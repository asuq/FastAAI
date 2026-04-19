#!/usr/bin/env python3
"""Command-line wrapper for FastAAI clustering."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_main():
    """Load the clustering entry point without importing the full package."""
    module_path = Path(__file__).resolve().parents[1] / "fastaai" / "cluster_fastaai.py"
    spec = importlib.util.spec_from_file_location("cluster_fastaai_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load clustering module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    sys.exit(load_main()())
