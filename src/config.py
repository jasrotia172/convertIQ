"""Central paths and configuration, independent of the working directory."""

from pathlib import Path
import json
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = Path(os.environ.get("CONVERTIQ_ARTIFACTS", ROOT / "artifacts"))
REPORTS = ROOT / "reports"
RAW_FILE = DATA / "raw" / "online_shoppers_intention.csv"


def settings(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs/default.yaml") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
