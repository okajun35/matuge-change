import threading

import pytest

from backend.jobs import gate as gate_module


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.setattr(gate_module, "_state", gate_module._GateState())
    monkeypatch.setattr(gate_module, "release_memory", lambda: None)


class TestMatteSlot:
    """One process-wide gate: `MATTE_MAX_WORKERS` has to bound both API paths together."""

    def test_only_one_matting_runs_at_a_time_by_default(self, monkeypatch):
        monkeypatch.delenv("MATTE_MAX_WORKERS", raising=False)
        peak, active, lock = [0], [0], threading.Lock()
        started, release = threading.Event(), threading.Event()

        def work():
            with gate_module.matte_slot():
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                started.set()
                release.wait(timeout=5)
                with lock:
                    active[0] -= 1

        first = threading.Thread(target=work)
        first.start()
        started.wait(timeout=5)
        second = threading.Thread(target=work)
        second.start()
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert peak[0] == 1

    def test_the_limit_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MATTE_MAX_WORKERS", "2")
        with gate_module.matte_slot():
            assert gate_module.active_mattings() == 1
            with gate_module.matte_slot():
                assert gate_module.active_mattings() == 2

    def test_the_slot_is_returned_when_the_work_raises(self, monkeypatch):
        monkeypatch.delenv("MATTE_MAX_WORKERS", raising=False)
        with pytest.raises(RuntimeError), gate_module.matte_slot():
            raise RuntimeError("matting exploded")

        assert gate_module.active_mattings() == 0
        with gate_module.matte_slot():  # not deadlocked on the leaked slot
            pass

    def test_memory_is_released_on_success_and_on_failure(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(gate_module, "release_memory", lambda: calls.append("released"))

        with gate_module.matte_slot():
            pass
        with pytest.raises(RuntimeError), gate_module.matte_slot():  # failures release too
            raise RuntimeError("boom")

        assert calls == ["released", "released"]
