"""Stop one recorded LLWM launcher group; never pkill Python or /dev/accel users."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
from nanochat.run_control import RunControl, linux_start_ticks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid-file',help='this host launcher PID manifest')
    parser.add_argument('--control-dir',help='shared run-control root, filesystem or gs://')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--launch-id', help='attempt ID; defaults to run-id')
    args=parser.parse_args()
    if not args.pid_file and not args.control_dir: parser.error('Need pid-file or shared control-dir')
    if args.control_dir:
        RunControl(args.control_dir,args.launch_id or args.run_id).stop()
        print('Published STOP for this run only')
    if args.pid_file:
        record=json.loads(Path(args.pid_file).read_text())
        if record['run_id']!=args.run_id or record['hostname']!=socket.gethostname():
            parser.error('PID manifest belongs to another run/host')
        pid=record['worker_pgid']
        ticks=linux_start_ticks(pid)
        if ticks is None or record['worker_start_ticks'] is None or ticks!=record['worker_start_ticks'] or os.getpgid(pid)!=pid:
            parser.error('Process is absent/reused or not the recorded group leader; refusing to signal')
        os.killpg(pid,signal.SIGTERM)
        print(f'Sent SIGTERM to verified run process group {pid}')


if __name__=='__main__': main()
