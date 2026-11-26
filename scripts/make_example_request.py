"""Write an API request using an actual dataset row."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import RAW_FILE, REPORTS, save_json
from src.data.schema import FEATURES, validate_frame
from src.models.schemas import SessionInput

frame = validate_frame(pd.read_csv(RAW_FILE), target=True, raw=True)
payload = SessionInput(**frame[FEATURES].iloc[0].to_dict()).model_dump()
save_json(REPORTS / "example_request.json", payload)
print(REPORTS / "example_request.json")
