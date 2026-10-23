"""Download the LendingClub loan-level files used by this project.

The four archives cover loans issued between 2007 and 2015 and are served
without authentication, so the whole study is reproducible from a clean
checkout. Raw files land in ``data/raw`` and are excluded from version
control; only derived tables and figures are committed.

Usage
-----
    python src/download_data.py
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://resources.lendingclub.com"

# archive name -> issuance window it covers
VINTAGES: dict[str, str] = {
    "LoanStats3a": "2007-2011",
    "LoanStats3b": "2012-2013",
    "LoanStats3c": "2014",
    "LoanStats3d": "2015",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CHUNK = 1 << 20  # 1 MiB
RETRIES = 3


def download(name: str, window: str, dest_dir: Path) -> Path:
    """Fetch one archive, skipping the transfer if it is already on disk."""
    url = f"{BASE_URL}/{name}.csv.zip"
    dest = dest_dir / f"{name}.csv.zip"

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {name} ({window}): already present, {dest.stat().st_size / 1e6:.1f} MB")
        return dest

    for attempt in range(1, RETRIES + 1):
        try:
            print(f"  {name} ({window}): downloading...", end="", flush=True)
            start = time.monotonic()
            with urllib.request.urlopen(url, timeout=120) as response:
                total = int(response.headers.get("Content-Length", 0))
                written = 0
                with dest.open("wb") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
            elapsed = time.monotonic() - start
            if total and written != total:
                raise OSError(f"truncated: {written} of {total} bytes")
            print(f" {written / 1e6:.1f} MB in {elapsed:.0f} s")
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f" failed ({exc})")
            dest.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise
            time.sleep(5 * attempt)

    raise RuntimeError("unreachable")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {RAW_DIR}")

    for name, window in VINTAGES.items():
        download(name, window, RAW_DIR)

    total = sum(p.stat().st_size for p in RAW_DIR.glob("*.zip"))
    print(f"\nDone. {len(list(RAW_DIR.glob('*.zip')))} archives, {total / 1e6:.0f} MB total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
