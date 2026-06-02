"""SessionLogger — persists per-turn stats and full-session chat logs."""

import datetime
import json
from pathlib import Path
from src.config import cfg


class SessionLogger:
    """Logs pipeline statistics and saves complete session transcripts.

    Usage::

        logger = SessionLogger()
        logger.log_turn({...})          # one JSONL line per pipeline run
        logger.save_session([...])      # full chat as .md file
    """

    def __init__(self, log_dir: str | Path | None = None):
        self.log_dir = Path(log_dir or cfg.logs_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(self, stats: dict):
        """Append one JSON line to ``chats.jsonl``."""
        log_file = self.log_dir / "chats.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

    def save_session(self, history: list[dict]):
        """Write the full chat log to a timestamped markdown file in journal."""
        if not history:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chatlog-{timestamp}.md"
        filepath = cfg.journal / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"# Chat Session Log - "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                )
                for entry in history:
                    role = entry["role"].upper()
                    content = entry["content"]
                    f.write(f"**{role}**: {content}\n\n")
        except Exception as e:
            print(f"Error saving session: {e}")
