from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_record(path: str | Path, record: dict) -> dict:
    """Append a hash-linked experiment record and return the stored block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = "0" * 64
    if path.exists() and path.stat().st_size:
        last = path.read_text(encoding="utf-8").strip().splitlines()[-1]
        previous_hash = json.loads(last)["hash"]
    block = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_hash": previous_hash,
        "record": record,
    }
    block["hash"] = _digest(block)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(block, sort_keys=True) + "\n")
    return block


def verify(path: str | Path) -> bool:
    previous_hash = "0" * 64
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        block = json.loads(line)
        claimed = block.pop("hash")
        if block["previous_hash"] != previous_hash or _digest(block) != claimed:
            return False
        previous_hash = claimed
    return True
