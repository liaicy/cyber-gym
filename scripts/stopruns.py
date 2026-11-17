# -*- coding: utf-8 -*-
"""Kill every examsociety run process, and verify they are gone.

`kill $(cat *.pid)` targets the launching shell, and `pkill -f` is a no-op in
git-bash on Windows; in both cases the python child survives and keeps writing
to the run directory with the OLD code. That has now corrupted three runs --
once producing a whole batch of rejected_invalid records against a stale CLI
path, after the directory had already been deleted and recreated.
"""
import subprocess
import sys
import time

PS = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
    "| Where-Object { $_.CommandLine -like '*examsociety*' } "
    "| ForEach-Object { $_.ProcessId }"
)


def running() -> list[str]:
    r = subprocess.run(["powershell", "-c", PS], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip().isdigit()]


def main() -> int:
    pids = running()
    if not pids:
        print("no examsociety processes running")
        return 0
    print("killing:", ", ".join(pids))
    subprocess.run(
        ["powershell", "-c", f"Stop-Process -Id {','.join(pids)} -Force"],
        capture_output=True,
    )
    for _ in range(10):
        time.sleep(1)
        if not (left := running()):
            print("all stopped")
            return 0
    print("STILL RUNNING:", ", ".join(left), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
