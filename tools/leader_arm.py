#!/usr/bin/env python3
"""Run the installed platform entry point from a source checkout."""

import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
os.environ.setdefault('XYC_PROJECT_ROOT', str(PROJECT))

if __name__ == '__main__':
    from xyc.teleop.calibrate import main
    raise SystemExit(main())
