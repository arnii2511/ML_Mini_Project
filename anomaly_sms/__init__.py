"""SMS anomaly detection package."""

from __future__ import annotations

import os


# Keeps joblib from probing system core details in restricted Windows environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
