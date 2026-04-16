"""Module entrypoint for ``python -m somehand.cli``."""

from __future__ import annotations

import sys

from .main import main


if __name__ == "__main__":
    main(sys.argv[1:])
