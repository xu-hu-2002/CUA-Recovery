#!/usr/bin/env python3
"""Platform shim: mdl launcher 只吃裸 .py entry，这里转发到同目录 .sh。"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "entry_derail_opencua_nebula.sh")
sys.exit(subprocess.call(["bash", script]))
