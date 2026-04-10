import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dex_mujoco.pico_input as pico_input


class _FakeThread:
    def __init__(self, target, daemon):
        self._started = False

    def start(self):
        self._started = True

    def is_alive(self):
        return self._started

    def join(self, timeout=None):
        return None


def test_pico_provider_shares_sdk_init_across_multiple_instances(monkeypatch):
    fake_xrt = SimpleNamespace(init_calls=0, close_calls=0)

    def _init():
        fake_xrt.init_calls += 1

    def _close():
        fake_xrt.close_calls += 1

    fake_xrt.init = _init
    fake_xrt.close = _close
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    monkeypatch.setattr(pico_input.threading, "Thread", _FakeThread)
    monkeypatch.setattr(pico_input, "_SDK_REFCOUNT", 0)

    left = pico_input.PicoHandProvider("left")
    right = pico_input.PicoHandProvider("right")

    assert fake_xrt.init_calls == 1

    left.close()
    assert fake_xrt.close_calls == 0

    right.close()
    assert fake_xrt.close_calls == 1


def test_pico_provider_reinitializes_sdk_after_all_instances_closed(monkeypatch):
    fake_xrt = SimpleNamespace(init_calls=0, close_calls=0)

    def _init():
        fake_xrt.init_calls += 1

    def _close():
        fake_xrt.close_calls += 1

    fake_xrt.init = _init
    fake_xrt.close = _close
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", fake_xrt)
    monkeypatch.setattr(pico_input.threading, "Thread", _FakeThread)
    monkeypatch.setattr(pico_input, "_SDK_REFCOUNT", 0)

    provider = pico_input.PicoHandProvider("left")
    provider.close()
    provider = pico_input.PicoHandProvider("right")
    provider.close()

    assert fake_xrt.init_calls == 2
    assert fake_xrt.close_calls == 2
