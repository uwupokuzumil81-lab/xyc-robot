"""Project paths shared by desktop entry points and learning tools."""

import os
from pathlib import Path


def project_root():
    if value := os.environ.get('XYC_PROJECT_ROOT'):
        return Path(value).expanduser().resolve()
    source = Path(__file__).resolve().parents[2]
    return source if (source / 'pyproject.toml').is_file() else Path.cwd().resolve()
