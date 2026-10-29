"""Rank-local, immediately flushed diagnostics; importing this does not initialize XLA."""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback

_logger = None
_handles = []


def enable_faults(directory, prefix=None, redirect=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = prefix or f'{socket.gethostname()}-pid{os.getpid()}'
    fault = (directory / f'{prefix}.fault.log').open('a', buffering=1)
    _handles.append(fault)
    faulthandler.enable(file=fault, all_threads=True)
    if hasattr(__import__('signal'), 'SIGUSR1'):
        import signal
        faulthandler.register(signal.SIGUSR1, file=fault, all_threads=True, chain=False)
    if redirect:
        fd = os.open(directory / f'{prefix}.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    return fault


class RankDiagnostics:
    def __init__(self, directory, rank=0, local_rank=0, run_id='local', debug=False, watchdog_seconds=0):
        self.host, self.rank, self.local_rank = socket.gethostname(), rank, local_rank
        self.run_id, self.debug = run_id, debug
        self.context = dict(step=None, microstep=None, pq_idx=None, rg_idx=None, epoch=None, batch_number=None)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f'{self.host}-rank{rank}-pid{os.getpid()}'
        self.file = (directory / f'{prefix}.jsonl').open('a', buffering=1)
        self.fault = enable_faults(directory, prefix)
        self.lock, self.closed = threading.Lock(), threading.Event()
        self.last_progress = time.monotonic()
        self.last_event = 'startup'
        self.watchdog_seconds = watchdog_seconds
        self.event('rank_start', python=sys.version, local_rank=local_rank)
        if watchdog_seconds > 0:
            threading.Thread(target=self._watchdog, name='llwm-stack-watchdog', daemon=True).start()

    def update(self, **fields):
        self.context.update(fields)

    def event(self, event, detail=False, **fields):
        # Even suppressed debug events advance the progress watchdog.
        self.last_progress, self.last_event = time.monotonic(), event
        if detail and not self.debug:
            return
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                      hostname=self.host, global_rank=self.rank, local_rank=self.local_rank,
                      pid=os.getpid(), run_id=self.run_id, **self.context)
        record.update(event=event, **fields)
        with self.lock:
            self.file.write(json.dumps(record, default=str, ensure_ascii=True) + '\n')
            self.file.flush()

    @contextmanager
    def span(self, stage, detail=False, **fields):
        start = time.monotonic()
        self.event(stage + '_begin', detail=detail, **fields)
        try:
            yield
        except StopIteration:
            self.event(stage + '_end', detail=detail, elapsed=time.monotonic()-start, end_of_group=True, **fields)
            raise
        except BaseException as exc:
            self.event(stage + '_error', elapsed=time.monotonic()-start,
                       exception=repr(exc), traceback=traceback.format_exc(), **fields)
            raise
        else:
            self.event(stage + '_end', detail=detail, elapsed=time.monotonic()-start, **fields)

    def _watchdog(self):
        while not self.closed.wait(min(30, self.watchdog_seconds)):
            elapsed = time.monotonic()-self.last_progress
            if elapsed >= self.watchdog_seconds:
                # This is a stack snapshot, not proof of deadlock or a kill timer.
                with self.lock:
                    self.fault.write(f'No Python progress for {elapsed:.1f}s; last={self.last_event}; context={self.context!r}. XLA compilation/execution may still be progressing.\n')
                    self.fault.flush()
                    faulthandler.dump_traceback(file=self.fault, all_threads=True)

    def close(self):
        self.closed.set()
        self.file.flush()
        self.fault.flush()


def configure(directory, rank=0, local_rank=0, run_id='local', debug=False, watchdog_seconds=0):
    global _logger
    if _logger:
        _logger.close()
    _logger = RankDiagnostics(directory, rank, local_rank, run_id, debug, watchdog_seconds)
    return _logger


def current():
    return _logger


def event(name, detail=False, **fields):
    if _logger:
        _logger.event(name, detail, **fields)


@contextmanager
def span(name, detail=False, **fields):
    if _logger:
        with _logger.span(name, detail, **fields):
            yield
    else:
        yield


def set_context(**fields):
    if _logger:
        _logger.update(**fields)


def xla_metrics(device):
    """Read runtime counters, without forcing an extra computation barrier."""
    if device.type != 'xla':
        return {}
    import torch_xla.debug.metrics as metrics
    import torch_xla.core.xla_model as xm
    result = {}
    for name in ('CompileTime', 'ExecuteTime', 'ExecuteReplicatedTime', 'TransferToDeviceTime', 'TransferFromDeviceTime'):
        values = metrics.metric_data(name)
        if values is not None:
            result[name] = dict(samples=values[0], accumulator=values[1])
    for name in ('CachedCompile', 'MarkStep', 'UncachedCompile'):
        value = metrics.counter_value(name)
        if value is not None:
            result[name] = value
    result['memory'] = xm.get_memory_info(device)
    try:
        import psutil
        result['host_rss_bytes'] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    return result
