from __future__ import annotations

import argparse
from typing import Protocol


class ExportBackend(Protocol):
    key: str
    description: str

    def parse_args(self, raw_argv: list[str]) -> argparse.Namespace:
        ...

    def run(self, args: argparse.Namespace, raw_argv: list[str]) -> None:
        ...
