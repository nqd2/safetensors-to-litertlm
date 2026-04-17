from __future__ import annotations

from typing import TextIO


class TeeTextIO:
    """Write to a primary stream (terminal) and a secondary stream (log file)."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        self._secondary.write(data)
        self._primary.flush()
        self._secondary.flush()
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    @property
    def encoding(self) -> str | None:  # pragma: no cover
        enc = getattr(self._primary, "encoding", None)
        return enc if isinstance(enc, str) else None

    def fileno(self) -> int:  # pragma: no cover
        return self._primary.fileno()

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        # absl.logging may call close() on sys.stderr at interpreter shutdown; do not
        # close the real terminal or the log file here (log_fp is closed in main()).
        self.flush()
