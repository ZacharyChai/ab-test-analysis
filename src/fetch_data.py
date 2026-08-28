"""Fetch the landing-page A/B test dataset into data/raw/.

The two CSVs are also committed to the repo, so a clean clone works offline and
the analysis is reproducible even if the mirror below goes away. This script
exists to document provenance and to let `make fetch` refresh the files.

Dataset: the widely-used Udacity "Analyze A/B Test Results" landing-page
experiment — 294,478 page views (2017-01-02 to 2017-01-24), each row a user
randomly assigned to the old or new page with a binary `converted` outcome,
plus a per-user country lookup.

Source mirror (raw GitHub):
  https://github.com/nirupamaprv/Analyze-AB-test-Results
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

BASE = "https://raw.githubusercontent.com/nirupamaprv/Analyze-AB-test-Results/master"
FILES = {
    "ab_data.csv": f"{BASE}/ab_data.csv",
    "countries.csv": f"{BASE}/countries.csv",
}

# sha256 of the files as committed — a fetch that doesn't match these is a
# signal the upstream mirror changed and the analysis should be re-checked.
EXPECTED_SHA256 = {
    "ab_data.csv": "d56e2accec25e99ac21cb3d76c5df516dd19cc7a77c14c9014f94e1ea1301beb",
    "countries.csv": "c011d0503d305c295327cd9dff9c37bb62a5f9ef4b356bcdaf7deca20d55e45b",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW_DIR / name
        print(f"fetching {name} <- {url}")
        urllib.request.urlretrieve(url, dest)
        digest = sha256(dest)
        print(f"  {dest.stat().st_size:,} bytes  sha256={digest}")
        expected = EXPECTED_SHA256.get(name)
        if expected and digest != expected:
            print(f"  WARNING: sha256 mismatch (expected {expected}) — "
                  f"upstream mirror may have changed; re-verify the analysis.")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
