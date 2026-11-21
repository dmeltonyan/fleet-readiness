"""Download NASA's C-MAPSS turbofan dataset into data/raw.

    python -m src.fetch_data
"""
from __future__ import annotations

import io
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NASA_ZIP = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
MIRROR = "https://github.com/edwardzjl/CMAPSSData.git"   # byte-identical copy, used if NASA is unreachable
EXPECTED = {"train_FD001.txt": 20631, "test_FD001.txt": 13096, "train_FD004.txt": 61249, "test_FD004.txt": 41214}


def _from_nasa() -> None:
    with urllib.request.urlopen(NASA_ZIP, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.endswith(".txt") and base[:3] in ("tra", "tes", "RUL"):
                (RAW / base).write_bytes(z.read(name))


def _from_mirror() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", MIRROR, tmp], check=True)
        for f in Path(tmp).glob("*_FD00*.txt"):
            (RAW / f.name).write_bytes(f.read_bytes())


def verify() -> None:
    for name, rows in EXPECTED.items():
        n = sum(1 for _ in open(RAW / name))
        assert n == rows, f"{name}: expected {rows} rows, found {n}"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        _from_nasa()
    except Exception as e:  # noqa: BLE001
        print(f"NASA download failed ({e}); using GitHub mirror")
        _from_mirror()
    verify()
    print(f"C-MAPSS ready in {RAW}")


if __name__ == "__main__":
    main()
