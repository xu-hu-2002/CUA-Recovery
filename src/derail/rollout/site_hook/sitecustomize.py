"""Process entry point for the DERAIL environment hook."""

import os
import sys

if os.environ.get("DERAIL_ENVIRONMENT_CONFIG") and os.path.basename(
    sys.argv[0] if sys.argv else ""
) == "run_mypcbench.py":
    try:
        from derail.rollout.state_probe import install_for_runner

        install_for_runner()
    except Exception as exc:
        print("[DERAIL env] failed to install environment hooks: %r" % (exc,), file=sys.stderr)
        raise SystemExit(2)
