"""NameDetector — extracts a user-provided name from natural language.

Supports both Spanish and English patterns.
"""

import re


class NameDetector:
    """Detects name declarations in user speech/text.

    Usage::

        detector = NameDetector()
        name = detector.detect("me llamo Marco")   # "Marco"
        name = detector.detect("my name is John")  # "John"
    """

    _PATTERNS = [
        r"(?:me\s+)?llamo\s+(\w+)",
        r"my name is (\w+)",
        r"(?:soy|sou)\s+(\w+)",
        r"call me (\w+)",
        r"(?:i'm|i am)\s+(\w+)",
    ]

    def detect(self, text: str) -> str | None:
        for pattern in self._PATTERNS:
            m = re.search(pattern, text.lower().strip())
            if m:
                return m.group(1).capitalize()
        return None
