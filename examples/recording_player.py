r"""
Compatibility launcher for ROIF Recording Player.

Run:

    python examples\recording_player.py output\run.roifrec

This launcher forwards all arguments to the root-level play.py.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from play import main


if __name__ == "__main__":
    raise SystemExit(main())
