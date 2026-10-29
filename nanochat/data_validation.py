"""Streaming, bounded text validation shared by preflight and training."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DataOptions:
    tokenizer_threads: int = 4
    tokenizer_batch_size: int = 128
    buffer_size: int = 1000
    max_buffer_tokens: int = 4_000_000
    max_document_chars: int = 1_000_000
    max_document_tokens: int = 262144
    max_tokenizer_chars: int = 1_000_000
    long_document_policy: str = 'error'
    packing_policy: str = 'legacy_crop'
    max_row_group_bytes: int = 0

    def __post_init__(self):
        for key in ('tokenizer_threads','tokenizer_batch_size','buffer_size','max_buffer_tokens',
                    'max_document_chars','max_document_tokens','max_tokenizer_chars'):
            if getattr(self,key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.long_document_policy not in ('error','chunk'):
            raise ValueError('long_document_policy must be error or chunk; data is never silently skipped')
        if self.packing_policy not in ('legacy_crop','retain_tail'):
            raise ValueError('packing_policy must be legacy_crop or retain_tail')
        if self.max_row_group_bytes < 0:
            raise ValueError('max_row_group_bytes must be nonnegative')

    def to_dict(self):
        return asdict(self)


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inventory(paths, tokenizer_files=()):
    import pyarrow.parquet as pq
    entries=[]
    for path in paths:
        path=Path(path)
        before=path.stat()
        pf=pq.ParquetFile(path)
        if 'text' not in pf.schema_arrow.names:
            raise ValueError(f'{path}: missing text column')
        groups=[dict(rows=pf.metadata.row_group(i).num_rows,
                     uncompressed_bytes=pf.metadata.row_group(i).total_byte_size)
                for i in range(pf.num_row_groups)]
        digest=sha256_file(path)
        if path.stat().st_size!=before.st_size or path.stat().st_mtime_ns!=before.st_mtime_ns:
            raise RuntimeError(f'Shard changed during inventory: {path}')
        entries.append(dict(name=path.name,size=before.st_size,sha256=digest,
                            rows=pf.metadata.num_rows,columns=pf.schema_arrow.names,
                            arrow_schema=str(pf.schema_arrow),row_groups=groups))
    tokenizers=[dict(name=Path(p).name,size=Path(p).stat().st_size,sha256=sha256_file(p)) for p in tokenizer_files]
    value=dict(shards=entries,tokenizer=tokenizers)
    return dict(version=1,**value,digest=canonical_digest(value))


def check_inventory(expected,actual):
    if expected['digest']!=actual['digest']:
        raise ValueError('Dataset/tokenizer inventory mismatch: names, ordering, metadata, size or SHA256 differ')


def validate_manifest(manifest,paths):
    """Cheap runtime guard; full content hashes are verified by preflight."""
    names=[Path(p).name for p in paths]
    if names != [s['name'] for s in manifest['shards']]:
        raise ValueError('Shard inventory/order changed since preflight')
    if not manifest.get('validation',{}).get('complete'):
        raise ValueError('Dataset manifest is not a successful full preflight; partial group probes are insufficient')
    for path,entry in zip(paths,manifest['shards']):
        if Path(path).stat().st_size!=entry['size']:
            raise ValueError(f'Shard size changed since preflight: {path}')
    stats=manifest.get('local_stats',{})
    for path in paths:
        saved=stats.get(Path(path).name)
        if saved and Path(path).stat().st_mtime_ns!=saved['mtime_ns']:
            raise ValueError(f'Shard modification time changed since preflight; rerun preflight: {path}')


def validate_ownership(manifest,world_size):
    shards=manifest['shards']
    if len(shards)<2:
        raise ValueError('Need at least one training shard and a separate validation shard')
    counts={}
    for split,entries in (('train',shards[:-1]),('val',shards[-1:])):
        owned=[sum(g['rows'] for entry in entries for i,g in enumerate(entry['row_groups']) if i%world_size==rank)
               for rank in range(world_size)]
        if min(owned)<=0:
            raise ValueError(f'At least one rank has no {split} rows: {owned}')
        counts[split]=owned
    return counts


def text_chunks(text, options, context):
    if not isinstance(text,str):
        raise ValueError(f'{context}: text must be a string, got {type(text).__name__}')
    limit=min(options.max_document_chars,options.max_tokenizer_chars)
    if len(text)>limit and options.long_document_policy=='error':
        raise ValueError(f'{context}: document has {len(text)} characters (effective limit {limit}); explicitly configure long-document policy')
    for start in range(0,max(1,len(text)),limit):
        chunk=text[start:start+limit]
        try:
            chunk.encode('utf-8',errors='strict')
        except UnicodeError as exc:
            raise ValueError(f'{context}: invalid UTF-8 text') from exc
        yield chunk


def tokenizer_batches(texts,options,context):
    batch,chars=[],0
    for row,text in enumerate(texts):
        for chunk in text_chunks(text,options,f'{context}, row={row}'):
            if batch and (len(batch)>=options.tokenizer_batch_size or chars+len(chunk)>options.max_tokenizer_chars):
                yield batch
                batch,chars=[],0
            batch.append(chunk)
            chars+=len(chunk)
    if batch:
        yield batch
