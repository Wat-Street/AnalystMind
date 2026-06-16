import json
from pathlib import Path

_TICKERS_PATH = Path(__file__).parents[4] / "tickers.json"


def load_tickers() -> list[str]:
    with _TICKERS_PATH.open() as f:
        return json.load(f)["tickers"]
