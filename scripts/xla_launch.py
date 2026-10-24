"""Launch a nanochat module on each TPU device via PJRT."""
import argparse
import os
import runpy
import sys


def _worker(index, module, argv):
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("module", help="e.g. scripts.base_train")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if os.environ["PJRT_DEVICE"] != "TPU":
        parser.error("xla_launch requires PJRT_DEVICE=TPU")
    if os.environ.get("XLA_USE_SPMD") == "1":
        parser.error("This launcher uses replicated data parallelism; unset XLA_USE_SPMD")
    import torch_xla
    torch_xla.launch(_worker, args=(args.module, args.args),
                     debug_single_process=args.single_process)


if __name__ == "__main__":
    main()
