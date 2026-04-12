"""Auto-chain training runs: wait for current run to finish, then start next.

Usage:
    python chain_runs.py                    # uses default queue
    python chain_runs.py --wait-pid 33020   # wait for specific PID first
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(Path(__file__).parent / "venv" / "Scripts" / "python.exe")

# Queue of (algo, epochs, config_path_or_None)
DEFAULT_QUEUE = [
    ("sac_v2", 400, "rl_training/configs/sac_v2_plastic_rr03.json"),
    ("sac_v2", 400, "rl_training/configs/sac_v2_plastic_rr1.json"),
    ("sac_v2", 400, "rl_training/configs/sac_v2_plastic.json"),
]


def pid_alive(pid):
    """Check if a process is still running."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def wait_for_pid(pid, poll_interval=60):
    print(f"Waiting for PID {pid} to finish...", flush=True)
    while pid_alive(pid):
        time.sleep(poll_interval)
    print(f"PID {pid} exited.", flush=True)


def run_training(algo, epochs, config_path=None):
    cmd = [PYTHON, "train.py", "--algo", algo, "--epochs", str(epochs)]
    if config_path:
        cmd += ["--config", config_path]
    print(f"\n{'='*60}")
    print(f"Starting: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None,
                        help="Wait for this PID to exit before starting queue")
    args = parser.parse_args()

    if args.wait_pid:
        wait_for_pid(args.wait_pid)

    for algo, epochs, config in DEFAULT_QUEUE:
        rc = run_training(algo, epochs, config)
        if rc != 0:
            print(f"WARNING: {algo} exited with code {rc}")

    print("\nAll queued runs complete.")


if __name__ == "__main__":
    main()
