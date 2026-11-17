#!/usr/bin/env python3
"""Run the project's training CLI from any working directory."""

import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
os.environ.setdefault('XYC_PROJECT_ROOT', str(PROJECT))
os.environ.setdefault('TORCH_HOME', str(PROJECT / '.setup/torch'))

if __name__ == '__main__':
    from xyc.learning.cli import main
    raise SystemExit(main())
