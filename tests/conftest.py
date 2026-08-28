"""Make `import experiment as ex` work from tests/ without installing a package.

run_analysis.py imports experiment the same bare way, relying on Python adding
the script's own directory to sys.path when run directly. Tests don't get that
for free, so this adds src/ explicitly, once, for the whole test session.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
