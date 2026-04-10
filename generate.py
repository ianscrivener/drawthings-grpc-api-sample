#!/usr/bin/env python3
"""Compatibility wrapper for the packaged generate CLI."""

import sys
import os
try:
    from drawthings_grpc_sample.generate import main
except ModuleNotFoundError:
    # Add src/ to sys.path if not already present
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from drawthings_grpc_sample.generate import main

if __name__ == "__main__":
    main(sys.argv[1:])
