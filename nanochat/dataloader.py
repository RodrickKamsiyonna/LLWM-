"""Deterministic streaming Parquet loader with bounded packing and exact rank recovery.

The default packing rule preserves the original best-fit cropping algorithm.
Every discarded suffix is counted. `retain_tail` is an explicit alternative.
XLA inputs are fresh immutable tensors, never views of a reused device buffer.
"""
from array import array
from collections import deque
import copy
from pathlib import Path
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.data_validation import DataOptions, canonical_digest, tokenizer_batches
from nanochat import diagnostics as diag


class ParquetTextStream:
    def __init__(self,paths,rank,world_size,batch_size,options,state=None):
        self.paths=list(paths)
        self.rank,self.world_size,self.batch_size=rank,world_size,batch_size
        self.options=options
        self.pq_idx,self.rg_idx,self.epoch,self.row_offset=0,rank,1,0
        self._iterator=None
        self.last_context={}
        if not paths:
            raise ValueError('No Parquet files for this split')
        # Fail rather than loop forever if a rank owns no rows in this split.
        if not any(any(pq.ParquetFile(p).metadata.row_group(i).num_rows for i in range(rank,pq.ParquetFile(p).num_row_groups,world_size)) for p in paths):
            raise ValueError(f'Rank {rank}/{world_size} owns no rows in this split')
        if state:
            for name in ('pq_idx','rg_idx','epoch','row_offset'):
                setattr(self,name,state[name])
            if self.rg_idx%world_size!=rank:
                raise ValueError('Checkpoint row group belongs to a different rank/topology')

    def state_dict(self):
        return {k:getattr(self,k) for k in ('pq_idx','rg_idx','epoch','row_offset')}

    def context(self):
        return dict(shard=str(self.paths[self.pq_idx]),pq_idx=self.pq_idx,rg_idx=self.rg_idx,
                    epoch=self.epoch,global_rank=self.rank,row_offset=self.row_offset)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            path=self.paths[self.pq_idx]
            if self._iterator is None:
                context=self.context()
                with diag.span('parquet_open',**context):
                    pf=pq.ParquetFile(path)
                    if 'text' not in pf.schema_arrow.names:
                        raise ValueError(f'{context}: missing text column')
                    if self.rg_idx>=pf.num_row_groups:
                        self.pq_idx+=1
                        self.rg_idx,self.row_offset=self.rank,0
                        if self.pq_idx>=len(self.paths):
                            self.pq_idx=0
                            self.epoch+=1
                        continue
                    size=pf.metadata.row_group(self.rg_idx).total_byte_size
                    if self.options.max_row_group_bytes and size>self.options.max_row_group_bytes:
                        raise ValueError(f'{context}: row group uncompressed bytes {size} exceeds configured limit')
                    self._iterator=iter(pf.iter_batches(batch_size=self.batch_size,row_groups=[self.rg_idx],columns=['text']))
                    # Replay just the current group to its exact row offset, without
                    # tokenizing or buffering its already-consumed prefix.
                    skip=self.row_offset
                    while skip:
                        batch=next(self._iterator)
                        if len(batch)>skip:
                            raise ValueError('Stored row offset is not a record-batch boundary; keep tokenizer batch size on stream replay')
                        skip-=len(batch)
            context=self.context()
            try:
                with diag.span('parquet_read',**context):
                    batch=next(self._iterator)
                    texts=batch.column(0).to_pylist()
            except StopIteration:
                self._iterator=None
                self.rg_idx+=self.world_size
                self.row_offset=0
                continue
            self.last_context=context
            self.row_offset+=len(texts)
            return texts,context


class LoaderState(dict):
    """Cheap progress view. snapshot() captures buffered documents only at save time."""
    def __init__(self,loader):
        super().__init__(loader.source.last_context or loader.source.context())
        self.update(batch_number=loader.batch_number,cropped_tokens=loader.cropped_tokens)
        self._loader=loader

    def snapshot(self):
        return self._loader.state_dict(replay_current=True)


class PackedDataLoader:
    VERSION=2

    def __init__(self,tokenizer,B,T,split,device='cpu',options=None,state=None,paths=None,rank=None,world_size=None):
        if split not in ('train','val') or B<=0 or T<=0:
            raise ValueError('Need train/val split and positive batch/context')
        _,current_rank,_,current_world=get_dist_info()
        self.rank=current_rank if rank is None else rank
        self.world_size=current_world if world_size is None else world_size
        if not 0<=self.rank<self.world_size:
            raise ValueError('Invalid global rank/world size')
        self.tokenizer,self.B,self.T,self.split=tokenizer,B,T,split
        self.vocab_size=tokenizer.get_vocab_size() if hasattr(tokenizer,"get_vocab_size") else 0x100000000
        self.device=torch.device(device)
        self.options=options or DataOptions()
        all_paths=list(paths) if paths is not None else list_parquet_files(warn_on_legacy=self.rank==0 and split=='train')
        if len(all_paths)<2:
            raise ValueError('Need separate training and validation shards')
        self.inventory=[dict(name=Path(p).name,size=Path(p).stat().st_size) for p in all_paths]
        selected=all_paths[:-1] if split=='train' else all_paths[-1:]
        self.doc_buffer=[]
        self.pending=deque()
        self.buffer_tokens=0
        self.batch_number=self.cropped_tokens=0
        self.current_cpu=None
        self.replay=None
        source_state=None
        if state:
            if state.get('version')!=self.VERSION:
                raise ValueError('Legacy loader state is approximate and cannot provide exact replay. Use an explicit legacy-resume override in the trainer, or start a new run.')
            expected=dict(rank=self.rank,world_size=self.world_size,B=B,T=T,split=split,inventory=self.inventory)
            for key,value in expected.items():
                if state[key]!=value:
                    raise ValueError(f'Data loader checkpoint mismatch: {key}')
            # Thread count changes are safe; batch size is kept fixed to replay a group offset.
            old=dict(state['options']); new=self.options.to_dict()
            old.pop('tokenizer_threads'); new.pop('tokenizer_threads')
            if old!=new:
                raise ValueError('Data policy/packing/tokenizer batch configuration differs from checkpoint')
            self.doc_buffer=copy.deepcopy(state['doc_buffer'])
            self.pending=deque(copy.deepcopy(state['pending']))
            self.buffer_tokens=sum(map(len,self.doc_buffer))
            self.batch_number=state['batch_number']
            self.cropped_tokens=state['cropped_tokens']
            self.replay=state.get('replay')
            source_state=state['source']
        self.source=ParquetTextStream(selected,self.rank,self.world_size,self.options.tokenizer_batch_size,self.options,source_state)

    def __iter__(self):
        return self

    def _tokenize(self):
        texts,context=next(self.source)
        bos=self.tokenizer.get_bos_token_id()
        for batch in tokenizer_batches(texts,self.options,context):
            with diag.span('tokenizer',batch_number=self.batch_number,documents=len(batch),
                           characters=sum(map(len,batch)),threads=self.options.tokenizer_threads,**context):
                result=self.tokenizer.encode(batch,prepend=bos,num_threads=self.options.tokenizer_threads)
                if len(result)!=len(batch):
                    raise ValueError(f'{context}: tokenizer returned wrong number of documents')
                for tokens in result:
                    if not tokens or any(not isinstance(t,int) or t<0 or t>=self.vocab_size or t>0xffffffff for t in tokens):
                        raise ValueError(f'{context}: invalid tokenizer output')
                    if len(tokens)>self.options.max_document_tokens:
                        if self.options.long_document_policy=='error':
                            raise ValueError(f'{context}: tokenized document length {len(tokens)} exceeds {self.options.max_document_tokens}')
                        limit=self.options.max_document_tokens
                        if limit<2:
                            raise ValueError('Token chunking needs max_document_tokens >= 2')
                        payload=tokens[1:] if tokens[0]==bos else tokens
                        for i in range(0,len(payload),limit-1):
                            self.pending.append(array('I',[bos,*payload[i:i+limit-1]]))
                    else:
                        self.pending.append(array('I',tokens))

    def _fill(self):
        while len(self.doc_buffer)<self.options.buffer_size:
            if not self.pending:
                self._tokenize()
            # Preserve the original refill granularity/order, including its
            # bounded overshoot past buffer_size by one tokenizer batch.
            while self.pending:
                doc=self.pending.popleft()
                if self.buffer_tokens+len(doc)>self.options.max_buffer_tokens:
                    raise MemoryError(f'Best-fit buffer would exceed {self.options.max_buffer_tokens} tokens; increase the explicit cap or reduce buffer_size (changes packing). No document was skipped.')
                self.doc_buffer.append(doc)
                self.buffer_tokens+=len(doc)

    def _transfer(self,inputs,targets):
        # Fresh host storage and fresh XLA handles. No subsequent batch mutates
        # this storage, even if the preceding lazy graph has not executed yet.
        self.current_cpu=(inputs,targets)
        with diag.span('input_transfer',detail=True,batch_number=self.batch_number,**(self.source.last_context or self.source.context())):
            x=inputs.to(self.device,non_blocking=self.device.type=='cuda')
            y=targets.to(self.device,non_blocking=self.device.type=='cuda')
        return x,y,LoaderState(self)

    def __next__(self):
        if self.replay is not None:
            inputs,targets=self.replay
            self.replay=None
            return self._transfer(inputs,targets)
        rows=torch.empty((self.B,self.T+1),dtype=torch.long)
        for row in range(self.B):
            position=0
            while position<self.T+1:
                self._fill()
                remaining=self.T+1-position
                best=-1
                length=0
                for i,doc in enumerate(self.doc_buffer):
                    if length<len(doc)<=remaining:
                        best,length=i,len(doc)
                if best<0:
                    best=min(range(len(self.doc_buffer)),key=lambda i:len(self.doc_buffer[i]))
                doc=self.doc_buffer.pop(best)
                self.buffer_tokens-=len(doc)
                take=min(len(doc),remaining)
                rows[row,position:position+take]=torch.tensor(doc[:take],dtype=torch.long)
                position+=take
                if take<len(doc):
                    if self.options.packing_policy=='retain_tail':
                        tail=array('I',[self.tokenizer.get_bos_token_id()])
                        tail.extend(doc[take:])
                        self.pending.appendleft(tail)
                    else:
                        removed=len(doc)-take
                        self.cropped_tokens+=removed
                        diag.event('packing_crop',detail=True,tokens_removed=removed,total_cropped_tokens=self.cropped_tokens,
                                   batch_number=self.batch_number,**self.source.last_context)
        self.batch_number+=1
        inputs,targets=rows[:,:-1].contiguous(),rows[:,1:].contiguous()
        if self.device.type=='cuda':
            inputs,targets=inputs.pin_memory(),targets.pin_memory()
        return self._transfer(inputs,targets)

    def state_dict(self,replay_current=True):
        value=dict(version=self.VERSION,rank=self.rank,world_size=self.world_size,B=self.B,T=self.T,
                   split=self.split,inventory=self.inventory,options=self.options.to_dict(),
                   source=self.source.state_dict(),doc_buffer=self.doc_buffer,pending=list(self.pending),
                   batch_number=self.batch_number,cropped_tokens=self.cropped_tokens,
                   replay=self.current_cpu if replay_current else None)
        return copy.deepcopy(value)


def _document_batches(split,resume_state_dict,tokenizer_batch_size):
    """Compatibility iterator with exact row-offset resume, never a group skip."""
    _,rank,_,world=get_dist_info()
    paths=list_parquet_files()
    paths=paths[:-1] if split=='train' else paths[-1:]
    source=ParquetTextStream(paths,rank,world,tokenizer_batch_size,DataOptions(tokenizer_batch_size=tokenizer_batch_size),resume_state_dict)
    for texts,context in source:
        yield texts,(context['pq_idx'],context['rg_idx'],context['epoch'])


def tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,B,T,split,tokenizer_threads=4,
        tokenizer_batch_size=128,device='cuda',resume_state_dict=None,buffer_size=1000,options=None):
    options=options or DataOptions(tokenizer_threads=tokenizer_threads,tokenizer_batch_size=tokenizer_batch_size,buffer_size=buffer_size)
    return PackedDataLoader(tokenizer,B,T,split,device,options,resume_state_dict)


def tokenizing_distributed_data_loader_bos_bestfit(*args,**kwargs):
    for inputs,targets,state in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args,**kwargs):
        yield inputs,targets
