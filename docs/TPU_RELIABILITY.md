# TPU v4-16 reliability qualification

Work is on `fix/tpu-v4-16-reliability`. Hardware qualification is still in progress.
Passing CPU tests or backend toy tests does not establish that the 1.3B training
crash is fixed. Keep the actual qualification logs with the run.

## Server storage

The current deployment uses server disks, without GCS credentials:

- Worker 0 (`10.130.0.2`) owns `/srv/llwm-checkpoints` on its boot disk.
- Worker 1 (`10.130.0.3`) mounts that directory using NFSv4.2 at the same path.
- The export allows only worker 1's private IP, with `sync` and `root_squash`.
- The training user has matching UID/GID 2002 on both hosts.
- Worker 1 has an `_netdev,nofail` fstab entry. Its unmounted directory is owned
  by root, so the training user cannot silently save to the wrong local disk.

Check both hosts before launching:

```bash
findmnt -T /srv/llwm-checkpoints
df -h /srv/llwm-checkpoints
```

The trainer also writes a probe from every rank and checks all probes are visible
before training. `CHECKPOINT_SHARED=1` is an assertion, not a substitute for that
check. Independent identically named local directories are rejected.

This is server storage, not a backup service. Worker 0 and its disk must remain
available for checkpoint/control operations. Do not delete/recreate that VM until
checkpoints have been copied elsewhere. A hard NFS mount can block on server
failure; local rank fault logs remain available, and the verified local PID-file
stop command below can stop the intended launcher independently of shared control.
Keep free disk for three approximately 10 GB complete checkpoints, temporary save
files and the per-host restore cache. Incomplete attempts are retained as evidence
and are not counted as usable checkpoints.

## Dataset preflight (on both hosts, as the training user)

From the checkout, with its TPU environment installed:

```bash
export NANOCHAT_BASE_DIR=/home/macbookpro/llwm_artifacts
export OMP_NUM_THREADS=1
python -m scripts.dataset_preflight --world-size=8 --workers=8 \
  --max-document-chars=16777216 --max-document-tokens=4194304 \
  --max-tokenizer-chars=16777216 \
  --output=evidence/full-manifest-large.json --log-dir=evidence/full-preflight-large
```

Run this independently on both hosts, compare the output `digest` values and
preserve both reports. They cover ordering, schema, row counts, byte sizes,
SHA256 shard/tokenizer checksums, valid strings, and actual Rust tokenization of
every group. `--compare=/path/to/other-report.json` also checks inventory equality.
Full manifests are required for distributed TPU training. The trainer checks the
manifest, current file sizes/mtimes, tokenizer hashes, policy settings and equal
cross-rank digests before model allocation.

To isolate the reported transition, add `--shard=0 --groups=56:72
--isolate-row-groups`, with separate output/log names. Repeat with
`--tokenizer-threads=1` and `4`. These partial probes cannot authorize training.
`--worker-timeout` and `--worker-memory-gb` can isolate native hangs/memory issues.
Linux `ru_maxrss` is recorded in KiB. Exceptions identify the exact shard, group,
row and rank; subprocess crashes retain fault/stdout logs. Nothing is skipped.

The strict generic document limit is one million characters; a real document in
shard 7/group 15 exceeds that limit. The v4 profile uses explicit larger guards
above, retaining whole documents. Policy `error` stops the run on an oversized,
malformed or incompatible document. Optional `chunk` must be explicitly selected
in preflight and training; it changes document boundaries and is not used by this
qualification. The original best-fit crop policy is preserved and its discarded
suffix count recorded. `retain_tail` is an explicit different packing policy.
Packed `uint32` document buffers are capped; exceeding the cap errors, never skips.

## Launch the original model

Use the same `RUN_ID`, `ATTEMPT_ID` and configuration on both hosts. Start each
host command in a persistent tmux session, as `macbookpro`. Supply each host's
successful full manifest (the current audit preflight runs in the separate
`/home/macbookpro/LLWM_reliability_20261029` checkout):

```bash
cd /home/macbookpro/LLWM_reliability_runtime_20261029
export NANOCHAT_BASE_DIR=/home/macbookpro/llwm_artifacts
export RUN_ID=llwm-d15-qualification-20261029
export ATTEMPT_ID=$RUN_ID-a1
export CHECKPOINT_ROOT=/srv/llwm-checkpoints
export DATASET_MANIFEST=/home/macbookpro/LLWM_reliability_20261029/evidence/full-manifest-large.json
export STOP_AFTER_STEP=300
export SKIP_SETUP=1
mkdir -p evidence
bash runs/tpu_v4_16.sh > evidence/$ATTEMPT_ID-host.log 2>&1
```

The profile preserves encoder 15, predictor 11, embedding width 1664 (aspect110,
head128), action width128/layers4, full attention (`L`), BF16, context1024,
device batch1, global131072 tokens, accumulation16, **6000 total optimizer steps**
and **786,432,000 intended tokens**. `STOP_AFTER_STEP=300` limits the qualification
attempt without shortening the schedule or changing weight-decay scaling.
All existing losses and weights remain unchanged. `SAVE_EVERY=25` and retention3
are defaults. Validation, CORE, samples, planning, SFT and post-evaluation are off.
Each interval accepts `-1` explicitly; it is not replaced with another value.
The actual parsed configuration and model configuration are printed in rank logs.

`NUM_ITERATIONS`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`, `EVAL_EVERY`,
`CORE_METRIC_EVERY`, `SAMPLE_EVERY`, `PLANNING_EVERY`, `SAVE_EVERY`, `MODEL_TAG`,
`TOKENIZER_THREADS` and `TOKENIZER_BATCH_SIZE` are independently configurable.
Changing training geometry/objectives/schedule after a checkpoint is rejected for
exact resume. Preview without touching devices using `DRY_RUN=1`.

## Monitor, stop and resume

```bash
tail -f evidence/$ATTEMPT_ID-host.log
find logs/$RUN_ID/$ATTEMPT_ID -name '*.jsonl' -o -name '*.stdout.log' -o -name '*.fault.log'
ls /srv/llwm-checkpoints/$RUN_ID/complete_*.json
```

Every rank has timestamped hostname/PID/rank/step/microstep/data-cursor diagnostics.
Native stdout/stderr is captured before importing XLA. `DEBUG_DIAGNOSTICS=1` adds
before/after microstep events. Normal mode keeps exception, checkpoint and periodic
XLA metrics. A configurable stack-dump watchdog diagnoses stopped progress without
calling a long compile a deadlock. `XLA_METRICS_EVERY=25` records compilation counts,
compilation/execution times, transfer metrics, device memory and host RSS.

Stop both hosts through the shared control namespace, from either host:

```bash
.venv/bin/python -m scripts.stop_run --run-id="$RUN_ID" --launch-id="$ATTEMPT_ID" \
  --control-dir=/srv/llwm-checkpoints/control
```

If shared storage is unavailable, run locally on each host:

```bash
.venv/bin/python -m scripts.stop_run --run-id="$RUN_ID" \
  --pid-file="logs/$RUN_ID/$ATTEMPT_ID/$(hostname)/launcher.json"
```

PID identity and process-group ownership are checked. No global `pkill`, device
kill, or unrelated workload termination is used. After stopping, verify with
`sudo fuser -v /dev/accel*` on **both** hosts. Record any remaining owners.
Stopping does not create an emergency checkpoint during an in-flight collective;
resume uses the last fully committed periodic save.

Resume on both hosts using the same run, manifest and 6000-step budget:

```bash
export ATTEMPT_ID=$RUN_ID-a2
export RESUME_LATEST=1
export STOP_AFTER_STEP=300
bash runs/tpu_v4_16.sh > evidence/$ATTEMPT_ID-host.log 2>&1
```

For a specific complete step, unset `RESUME_LATEST` and set
`RESUME_FROM_STEP=150`. A fresh attempt ID avoids old STOP/failure records.
Checkpoint manifests commit only after all ranks report successful writes.
The replicated XLA model/optimizer are saved once, alongside every rank's exact
buffered loader state, prefetched batch, RNG and loop/scheduler state. SHA256 and
size checks reject corrupt/incomplete files. CUDA optimizer shards remain sharded.
Old approximate loader checkpoints cannot provide exact recovery; legacy weight
loading remains available. Retention protects the latest two complete checkpoints
and never deletes legacy files or another run's artifacts.

## Required real hardware tests

Launch the following command on **both hosts**, under the supervised launcher,
with a fresh run/attempt ID for each test:

```bash
.venv/bin/python -m scripts.xla_launch --run-id=backend16384 --launch-id=backend16384-a1 \
  --control-dir=/srv/llwm-checkpoints/control --expected-world-size=8 \
  --expected-local-devices=4 --debug scripts.tpu_reliability_test \
  --global-tokens=16384 --steps=300 --run-id=backend16384 \
  --checkpoint-dir=/srv/llwm-checkpoints/backend16384 --checkpoint-shared
```

Repeat with global131072 and a different ID/destination. These backend regression
tests require all eight devices and two distinct hosts; check analytical averaged
gradients, independent AdamW updates, equal replicas, restore model and optimizer
moments, and bound post-warmup compilation. They do not replace the original model.
Test failure cleanup using another fresh ID and
`--inject-failure-rank=5 --inject-failure-step=3 --native-failure`; expect a nonzero
exit on both hosts, preserved fault logs and released accelerator handles.

Then run the original profile to at least300, save before183, resume across both
hosts and repeat crossing183. Preserve throughput and XLA metrics. Compare
`XLA_SYNC_MODE=microstep` (asynchronous graph boundaries) against diagnostic
`microstep-wait`; `step` is available to diagnose whole-accumulation graph growth.
Only change execution strategy after checking gradient parity and throughput.

Run evaluation independently from a complete checkpoint on both hosts:

```bash
.venv/bin/python -m scripts.xla_launch --run-id=eval-qualification --launch-id=eval-a1 \
  --control-dir=/srv/llwm-checkpoints/control --expected-world-size=8 \
  --expected-local-devices=4 scripts.base_eval --device-type=xla \
  --checkpoint-dir=/srv/llwm-checkpoints/llwm-d15-qualification-20261029 \
  --eval=sample --eval-max-new-tokens=4 --unconditioned-max-new-tokens=4 \
  --plan-goal=. --plan-iterations=2 --device-batch-size=1
```

Follow with `--eval=bpb` and `--eval=core` as separate runs; CORE requires its task
bundle on each host. Short prompts use nonnegative slicing; goal remains `.` and
planning horizon is the allowed generated-token count. Expensive evaluation can
compile new shapes; a failure cannot invalidate an earlier complete checkpoint.
