"""Atomic filesystem or GCS artifacts. GCS uses object generations, not renames."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


class ArtifactStore:
    def __init__(self,root):
        self.root=str(root).rstrip('/')
        self.cloud=self.root.startswith('gs://')
        self.bucket=None
        if self.cloud:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError('GCS checkpoints/control require pip install google-cloud-storage or the cloud extra') from exc
            bucket,_,prefix=self.root[5:].partition('/')
            self.bucket=storage.Client().bucket(bucket)
            self.prefix=prefix.rstrip('/')
        else:
            Path(self.root).mkdir(parents=True,exist_ok=True)

    def _blob(self,name):
        return self.bucket.blob('/'.join(filter(None,(self.prefix,name))))

    def exists(self,name):
        return self._blob(name).exists() if self.cloud else (Path(self.root)/name).exists()

    def list(self,prefix=''):
        if self.cloud:
            full='/'.join(filter(None,(self.prefix,prefix)))
            base=len(self.prefix)+1 if self.prefix else 0
            return sorted(blob.name[base:] for blob in self.bucket.client.list_blobs(self.bucket,prefix=full))
        return sorted(str(p.relative_to(self.root)) for p in Path(self.root).rglob('*') if p.is_file() and str(p.relative_to(self.root)).startswith(prefix))

    def put_file(self,name,source,exclusive=False):
        if self.cloud:
            self._blob(name).upload_from_filename(str(source),if_generation_match=0 if exclusive else None,checksum='auto')
        else:
            target=Path(self.root)/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if exclusive and target.exists():
                raise FileExistsError(str(target))
            temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
            try:
                with open(source,'rb') as src,temporary.open('xb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst,8*1024*1024)
                    dst.flush(); os.fsync(dst.fileno())
                if exclusive:
                    os.link(temporary,target)  # create-if-absent, same filesystem
                    temporary.unlink()
                else:
                    os.replace(temporary,target)
                fd=os.open(target.parent,os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
            finally:
                if temporary.exists(): temporary.unlink()

    def put_json(self,name,value,exclusive=False):
        raw=json.dumps(value,sort_keys=True,ensure_ascii=True).encode()
        if self.cloud:
            self._blob(name).upload_from_string(raw,content_type='application/json',if_generation_match=0 if exclusive else None)
        else:
            with tempfile.NamedTemporaryFile() as f:
                f.write(raw); f.flush()
                self.put_file(name,f.name,exclusive)

    def get_json(self,name):
        if self.cloud:
            return json.loads(self._blob(name).download_as_bytes())
        return json.loads((Path(self.root)/name).read_text())

    def get_file(self,name,target):
        if self.cloud:
            self._blob(name).download_to_filename(str(target),checksum='auto')
        else:
            import shutil
            shutil.copyfile(Path(self.root)/name,target)

    def delete(self,name):
        if self.cloud:
            blob=self._blob(name)
            blob.reload()
            blob.delete(if_generation_match=blob.generation)
        else:
            (Path(self.root)/name).unlink(missing_ok=True)

    def wait_json(self,name,timeout=600):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.exists(name): return self.get_json(name)
            time.sleep(1)
        raise TimeoutError(f'Timed out waiting for {self.root}/{name}')


def file_record(path,name):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return dict(name=name,size=Path(path).stat().st_size,sha256=h.hexdigest())
