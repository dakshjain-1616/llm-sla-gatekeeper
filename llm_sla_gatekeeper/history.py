"""
Validation history tracking — appends results to a JSONL file for trend analysis.

Each record is one JSON line (JSONL format) so the file is streamable and append-safe.

Usage:
    from llm_sla_gatekeeper.history import append_result, load_history, history_for_model, clear_history
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_FILE = Path(os.getenv("SLA_HISTORY_FILE", "outputs/history.jsonl"))


def append_result(
    result_dict: dict,
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> None:
    """Append one ValidationResult dict as a single JSON line to the history file.

    Creates parent directories if they don't exist. Silently skips on write errors
    so a history failure never interrupts the main validation flow.
    """
    try:
        history_file = Path(history_file)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with history_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result_dict, separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.warning("Could not write to history file %s: %s", history_file, exc)


def load_history(
    history_file: Path = DEFAULT_HISTORY_FILE,
    limit: int = 200,
) -> List[dict]:
    """Load the most recent *limit* records from the JSONL history file.

    Returns an empty list if the file doesn't exist or is unreadable.
    """
    history_file = Path(history_file)
    if not history_file.exists():
        return []
    try:
        lines = history_file.read_text(encoding="utf-8").splitlines()
        records: List[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.debug("Skipping malformed history line: %s", exc)
        return records[-limit:]
    except OSError as exc:
        logger.warning("Could not read history file %s: %s", history_file, exc)
        return []


def history_for_model(
    model_path: str,
    history_file: Path = DEFAULT_HISTORY_FILE,
    limit: int = 50,
) -> List[dict]:
    """Return history records for a specific model path (most recent last)."""
    all_records = load_history(history_file, limit=limit * 20)
    return [r for r in all_records if r.get("model_path") == model_path][-limit:]


def history_summary(history_file: Path = DEFAULT_HISTORY_FILE) -> dict:
    """Return aggregate stats across all history records.

    Returns a dict with keys: total, pass_count, fail_count, error_count,
    models_seen (list of unique model paths).
    """
    records = load_history(history_file, limit=0)  # load all
    summary: dict = {
        "total": len(records),
        "pass_count": sum(1 for r in records if r.get("status") == "PASS"),
        "fail_count": sum(1 for r in records if r.get("status") == "FAIL"),
        "error_count": sum(1 for r in records if r.get("status") == "ERROR"),
        "models_seen": list(dict.fromkeys(r.get("model_path", "") for r in records)),
    }
    return summary


def clear_history(history_file: Path = DEFAULT_HISTORY_FILE) -> int:
    """Delete the history file. Returns the number of records that were in it."""
    history_file = Path(history_file)
    if not history_file.exists():
        return 0
    count = len(load_history(history_file, limit=0))
    history_file.unlink()
    return count
