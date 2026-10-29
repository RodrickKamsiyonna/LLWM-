"""Validate Parquet/tokenizer inputs without constructing or importing a model.

Run on every TPU VM host. Compare the resulting inventories across hosts before
training. --groups 56:72 --shard 0 isolates the reported transition; only a full
preflight produces a manifest accepted by --dataset-manifest.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from nanochat.data_validation import DataOptions, inventory, check_inventory, validate_ownership, tokenizer_batches


def validate_group(path,group,tokenizer,options,pq_idx=0):
    import pyarrow.parquet as pq
    from nanochat import diagnostics as diag
    pf=pq.ParquetFile(path)
    context=dict(shard=str(path),pq_idx=pq_idx,rg_idx=group,epoch=1,global_rank=group%int(os.environ.get('PREFLIGHT_WORLD_SIZE','1')))
    if 'text' not in pf.schema_arrow.names:
        raise ValueError(f'{context}: missing text column')
    if options.max_row_group_bytes and pf.metadata.row_group(group).total_byte_size>options.max_row_group_bytes:
        raise ValueError(f'{context}: row group exceeds configured byte limit')
    rows=tokens=maximum=batch_number=0
    iterator=iter(pf.iter_batches(batch_size=options.tokenizer_batch_size,row_groups=[group],columns=['text']))
    while True:
        with diag.span('preflight_parquet_read',batch_number=batch_number,row_offset=rows,**context):
            try: batch=next(iterator)
            except StopIteration: break
            texts=batch.column(0).to_pylist()
        rows+=len(texts)
        batch_number+=1
        for values in tokenizer_batches(texts,options,context):
            with diag.span('preflight_tokenizer',batch_number=batch_number,characters=sum(map(len,values)),**context):
                encoded=tokenizer.encode(values,prepend=tokenizer.get_bos_token_id(),num_threads=options.tokenizer_threads)
                if len(encoded)!=len(values): raise ValueError(f'{context}: tokenizer document-count mismatch')
                for doc in encoded:
                    if not doc or any(not isinstance(t,int) or t<0 or t>=tokenizer.get_vocab_size() for t in doc):
                        raise ValueError(f'{context}: invalid token ids')
                    if len(doc)>options.max_document_tokens and options.long_document_policy=='error':
                        raise ValueError(f'{context}: document has {len(doc)} tokens; configured limit {options.max_document_tokens}')
                    tokens+=len(doc); maximum=max(maximum,len(doc))
    import resource
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(group=group,rows=rows,tokens=tokens,max_document_tokens=maximum,peak_rss_native_units=rss)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--output',default='dataset-manifest.json')
    parser.add_argument('--compare',help='manifest from the other host, local path or gs:// object')
    parser.add_argument('--world-size',type=int,default=8)
    parser.add_argument('--shard',type=int)
    parser.add_argument('--groups',help='half-open row-group range, e.g. 56:72')
    parser.add_argument('--tokenizer-threads',type=int,default=4)
    parser.add_argument('--tokenizer-batch-size',type=int,default=128)
    parser.add_argument('--max-document-chars',type=int,default=1_000_000)
    parser.add_argument('--max-document-tokens',type=int,default=262144)
    parser.add_argument('--max-tokenizer-chars',type=int,default=1_000_000)
    parser.add_argument('--long-document-policy',choices=['error','chunk'],default='error')
    parser.add_argument('--max-row-group-bytes',type=int,default=0)
    parser.add_argument('--isolate-row-groups',action='store_true',help='one subprocess per group to isolate native tokenizer faults')
    parser.add_argument('--workers',type=int,default=1,help='parallel isolated shard/group subprocesses per host')
    parser.add_argument('--worker-timeout',type=float,default=1800)
    parser.add_argument('--worker-memory-gb',type=float,default=0,help='Linux worker RLIMIT_AS, 0 disables')
    parser.add_argument('--log-dir',default='preflight-logs')
    parser.add_argument('--worker-spec',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.world_size<1: parser.error('world-size must be positive')
    if args.worker_spec:
        spec=json.loads(Path(args.worker_spec).read_text())
        if spec['memory_gb']:
            import resource
            limit=int(spec['memory_gb']*1024**3)
            resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        from nanochat.diagnostics import configure
        configure(spec['log_dir'],rank=spec['shard_index'],run_id='dataset-preflight',debug=True,watchdog_seconds=300)
        from nanochat.tokenizer import get_tokenizer
        tokenizer=get_tokenizer()
        options=DataOptions(**spec['options'])
        os.environ['PREFLIGHT_WORLD_SIZE']=str(spec['world_size'])
        results=[]
        for group in spec['groups']:
            print(json.dumps(dict(event='group_begin',shard=spec['path'],pq_idx=spec['shard_index'],rg_idx=group,rank=group%spec['world_size'])),flush=True)
            result=validate_group(spec['path'],group,tokenizer,options,spec['shard_index'])
            results.append(result)
            print(json.dumps(dict(event='group_complete',**result)),flush=True)
        Path(spec['result']).write_text(json.dumps(results))
        return
    from nanochat.diagnostics import configure
    configure(args.log_dir,run_id='preflight-parent',debug=True,watchdog_seconds=300)
    from nanochat.dataset import list_parquet_files
    from nanochat.common import get_base_dir
    paths=list_parquet_files(args.data_dir)
    token_dir=Path(get_base_dir())/'tokenizer'
    tokenizer_files=sorted(p for p in token_dir.iterdir() if p.is_file() and p.suffix in ('.pkl','.pt'))
    if not tokenizer_files: parser.error('No tokenizer artifacts found')
    manifest=inventory(paths,tokenizer_files)
    manifest['ownership_rows']=validate_ownership(manifest,args.world_size)
    if args.compare:
        if args.compare.startswith('gs://'):
            from nanochat.storage import ArtifactStore
            root,name=args.compare.rsplit('/',1)
            other=ArtifactStore(root).get_json(name)
        else: other=json.loads(Path(args.compare).read_text())
        check_inventory(other,manifest)
    options=DataOptions(tokenizer_threads=args.tokenizer_threads,tokenizer_batch_size=args.tokenizer_batch_size,
                        max_document_chars=args.max_document_chars,max_document_tokens=args.max_document_tokens,
                        max_tokenizer_chars=args.max_tokenizer_chars,long_document_policy=args.long_document_policy,
                        max_row_group_bytes=args.max_row_group_bytes)
    logs=Path(args.log_dir).resolve(); logs.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); reports=[]
    selected=range(len(paths)) if args.shard is None else [args.shard]
    tasks=[]
    for index in selected:
        groups=list(range(len(manifest['shards'][index]['row_groups'])))
        if args.groups:
            lo,hi=map(int,args.groups.split(':')); groups=[g for g in groups if lo<=g<hi]
        units=[[group] for group in groups] if args.isolate_row_groups else [groups]
        tasks.extend((index,unit) for unit in units)
    def check_unit(task):
        index,unit=task
        name=f'shard{index}-groups{unit[0]}-{unit[-1]}' if unit else f'shard{index}-empty'
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'result.json'; spec=Path(tmp)/'spec.json'
            spec.write_text(json.dumps(dict(path=str(paths[index]),shard_index=index,groups=unit,
                result=str(result),options=asdict(options),memory_gb=args.worker_memory_gb,
                world_size=args.world_size,log_dir=str(logs))))
            with (logs/f'{name}.stdout.log').open('w') as output:
                env=dict(os.environ,PYTHONFAULTHANDLER='1')
                env.pop('PJRT_DEVICE',None)  # preflight is always a CPU workload
                try:
                    process=subprocess.run([sys.executable,'-u','-m','scripts.dataset_preflight','--worker-spec',str(spec)],
                                           env=env,stdout=output,stderr=subprocess.STDOUT,timeout=args.worker_timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f'Preflight timeout: {paths[index]}, groups {unit}; inspect {output.name}') from exc
            if process.returncode:
                raise RuntimeError(f'Native/Python preflight failure exit={process.returncode}: {paths[index]}, groups={unit}; inspect {logs}')
            return dict(shard=index,groups=json.loads(result.read_text()))
    if args.workers<1: parser.error('workers must be positive')
    from concurrent.futures import ThreadPoolExecutor,as_completed
    executor=ThreadPoolExecutor(max_workers=args.workers)
    futures=[executor.submit(check_unit,task) for task in tasks]
    try:
        for future in as_completed(futures): reports.append(future.result())
    finally:
        for future in futures: future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
    reports.sort(key=lambda item:(item['shard'],item['groups'][0]['group'] if item['groups'] else -1))
    manifest.update(hostname=socket.gethostname(),validation=dict(complete=args.shard is None and args.groups is None,
                    options=asdict(options),reports=reports,elapsed=time.monotonic()-start),
                    local_stats={Path(p).name:dict(mtime_ns=Path(p).stat().st_mtime_ns) for p in paths})
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp'); temporary.write_text(json.dumps(manifest,indent=2)); os.replace(temporary,output)
    print(f"Preflight passed on {manifest['hostname']}: digest={manifest['digest']}; full={manifest['validation']['complete']}; output={output}")


if __name__=='__main__':
    main()
