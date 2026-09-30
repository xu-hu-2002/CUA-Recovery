#!/usr/bin/env python3
"""Entry shim that forwards to the sibling .sh script."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "entry_derail_opencua_nebula.sh")
sys.exit(subprocess.call(["bash", script]))
