#!/usr/bin/env python
"""RealMan LeRobot RTC-SFT entrypoint using the vendored OpenPI trainer."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
OPENPI_ROOT = ROOT / 'third_party' / 'openpi_rtc'


def main():
    # Select the matching model, data transforms and trainer together. Do not
    # accidentally import a different editable OpenPI installation in the venv.
    sys.path[:0] = [
        str(ROOT),
        str(OPENPI_ROOT / 'src'),
        str(OPENPI_ROOT / 'packages/openpi-client/src'),
    ]
    os.chdir(OPENPI_ROOT)
    runpy.run_path(str(OPENPI_ROOT / 'scripts/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
