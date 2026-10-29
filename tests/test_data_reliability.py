from dataclasses import replace
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat.data_validation import DataOptions,inventory,check_inventory,validate_ownership,text_chunks
from nanochat.dataloader import PackedDataLoader,ParquetTextStream


class Tokenizer:
    def get_bos_token_id(self): return 0
    def get_vocab_size(self): return 256
    def encode(self,texts,prepend=None,num_threads=1):
        return [[prepend]+[ord(ch)%255+1 for ch in text] for text in texts]


@pytest.fixture
def shards(tmp_path):
    paths=[]
    for shard in range(3):
        path=tmp_path/f'shard_{shard:05d}.parquet'
        with pq.ParquetWriter(path,pa.schema([('text',pa.string())])) as writer:
            for group in range(80):
                writer.write_table(pa.table({'text':[f'doc {shard} {group} {row}' for row in range(5)]}))
        paths.append(path)
    return paths


def options(**kwargs):
    return DataOptions(tokenizer_batch_size=2,buffer_size=4,max_buffer_tokens=10000,**kwargs)


@pytest.mark.parametrize('rank',range(8))
def test_groups_56_to_71_and_rank_stride(shards,rank):
    source=ParquetTextStream(shards[:1],rank,8,2,options())
    contexts=[];texts=[]
    while source.epoch==1:
        values,context=next(source)
        if context['epoch']!=1: break
        contexts.append(context['rg_idx']); texts.extend(values)
        encoded=Tokenizer().encode(values,prepend=0,num_threads=1)
        assert len(encoded)==len(values)
    assert set(contexts)==set(range(rank,80,8))
    assert {56+rank,64+rank}.issubset(contexts)
    assert len(texts)==50


def test_stream_crosses_shards_and_epoch_without_loss(shards):
    source=ParquetTextStream(shards[:2],3,8,2,options())
    values=[]
    for _ in range(61):
        texts,context=next(source);values.extend(texts)
    assert source.epoch==2
    assert len(set(values[:100]))==100


def test_input_storage_is_not_reused(shards):
    loader=PackedDataLoader(Tokenizer(),1,8,'train',paths=shards,options=options(),rank=0,world_size=8)
    x,y,_=next(loader); expected=x.clone(); expected_y=y.clone()
    for _ in range(20): next(loader)
    torch.testing.assert_close(x,expected);torch.testing.assert_close(y,expected_y)


@pytest.mark.parametrize('rank',range(8))
def test_exact_resume_including_prefetched_batch_and_buffer(shards,rank):
    loader=PackedDataLoader(Tokenizer(),1,8,'train',paths=shards,options=options(),rank=rank,world_size=8)
    for _ in range(19): current=next(loader)
    state=current[2].snapshot()
    assert state['doc_buffer'] and state['replay']
    reference=[(current[0].clone(),current[1].clone())]
    reference.extend((x.clone(),y.clone()) for x,y,_ in [next(loader) for _ in range(100)])
    restored=PackedDataLoader(Tokenizer(),1,8,'train',paths=shards,options=options(),state=state,rank=rank,world_size=8)
    for x,y in reference:
        actual_x,actual_y,_=next(restored)
        torch.testing.assert_close(actual_x,x);torch.testing.assert_close(actual_y,y)
    with pytest.raises(ValueError,match='rank'):
        PackedDataLoader(Tokenizer(),1,8,'train',paths=shards,options=options(),state=state,rank=(rank+1)%8,world_size=8)


def test_inventory_checks_hashes_order_and_metadata(shards,tmp_path):
    original=inventory(shards)
    copied=[]
    for path in shards:
        dest=tmp_path/'host2'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes());copied.append(dest)
    check_inventory(original,inventory(copied))
    with pytest.raises(ValueError,match='mismatch'):check_inventory(original,inventory(list(reversed(copied))))
    pq.write_table(pa.table({'text':['changed']}),copied[0])
    with pytest.raises(ValueError,match='mismatch'):check_inventory(original,inventory(copied))
    assert validate_ownership(original,8)['train']==[100]*8


def test_invalid_text_and_long_documents_are_explicit_errors():
    for value in (None,123,b'bytes'):
        with pytest.raises(ValueError,match='string'):list(text_chunks(value,options(),{}))
    with pytest.raises(ValueError,match='UTF-8'):list(text_chunks('\ud800',options(),{}))
    strict=options(max_document_chars=4)
    with pytest.raises(ValueError,match='characters'):list(text_chunks('abcdefgh',strict,{}))
    assert ''.join(text_chunks('abcdefgh',replace(strict,long_document_policy='chunk'),{}))=='abcdefgh'


def test_no_rows_for_rank_is_rejected_not_infinite_loop(tmp_path):
    path=tmp_path/'one.parquet';pq.write_table(pa.table({'text':['one']}),path)
    with pytest.raises(ValueError,match='no rows'):ParquetTextStream([path],7,8,2,options())


def test_default_packing_matches_original_best_fit(shards):
    # Independent original algorithm: batch-granularity refill, first largest
    # fitting document, otherwise crop the shortest. Compare actual token rows.
    tokenizer=Tokenizer()
    config=options()
    source=ParquetTextStream(shards[:-1],0,8,config.tokenizer_batch_size,config)
    pool=[]
    loader=PackedDataLoader(tokenizer,2,32,'train',paths=shards,options=config,rank=0,world_size=8)
    for _ in range(100):
        rows=[]
        for row in range(2):
            packed=[]
            while len(packed)<33:
                while len(pool)<config.buffer_size:
                    texts,_=next(source)
                    pool.extend(tokenizer.encode(texts,prepend=0,num_threads=4))
                remaining=33-len(packed)
                fitting=[i for i,doc in enumerate(pool) if len(doc)<=remaining]
                index=max(fitting,key=lambda i:len(pool[i])) if fitting else min(range(len(pool)),key=lambda i:len(pool[i]))
                packed.extend(pool.pop(index)[:remaining])
            rows.append(packed)
        expected=torch.tensor(rows)
        x,y,_=next(loader)
        torch.testing.assert_close(x,expected[:,:-1])
        torch.testing.assert_close(y,expected[:,1:])
