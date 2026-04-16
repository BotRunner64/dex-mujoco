import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


@pytest.mark.parametrize(
    "module_name",
    [
        "somehand.runtime.sink_outputs",
        "somehand.runtime.sink_rendering",
        "somehand.runtime.source_adapters",
        "somehand.runtime.source_recording",
        "somehand.runtime.source_transforms",
    ],
)
def test_runtime_modules_import_cleanly_in_fresh_interpreter(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                f"sys.path.insert(0, {str(SRC)!r}); "
                f"importlib.import_module({module_name!r})"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
