"""MatSense closed-loop campaign: operator, runtime adapter, scenarios, analysis."""
from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent


def code_sha256() -> str:
    """Hash of every .py file of the campaign, written into each run summary."""
    h = hashlib.sha256()
    files = sorted(list(PACKAGE_DIR.glob("*.py")) + list(ROOT_DIR.glob("*.py")))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()
