"""Download the UCI archive or validate and copy a manually obtained CSV."""

import argparse
from io import BytesIO
from pathlib import Path
import urllib.request
from zipfile import ZipFile
import hashlib
import pandas as pd
from src.config import RAW_FILE, REPORTS, save_json
from src.data.schema import validate_frame, quality_report

URL = "https://archive.ics.uci.edu/static/public/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip"


def download(manual: str | None = None, output: Path = RAW_FILE) -> Path:
    if manual:
        payload = Path(manual).read_bytes()
    else:
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "ConvertIQ/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                archive = ZipFile(BytesIO(response.read()))
            member = next(
                name for name in archive.namelist() if name.endswith("online_shoppers_intention.csv")
            )
            payload = archive.read(member)
        except Exception as exc:
            raise RuntimeError(
                "UCI download failed. Download the CSV manually and run "
                "python -m src.data.download --csv /path/to/file.csv"
            ) from exc
    frame = validate_frame(pd.read_csv(BytesIO(payload)), target=True, raw=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    save_json(REPORTS / "data_quality.json", quality_report(frame))
    save_json(
        REPORTS / "dataset_source.json",
        {
            "url": URL,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(frame),
            "source": "manual" if manual else "UCI",
            "license": "CC BY 4.0",
            "doi": "10.24432/C5F88Q",
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Manually downloaded UCI CSV")
    args = parser.parse_args()
    print(download(args.csv))
