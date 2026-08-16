"""Which solve mode a deployment is running has to be visible in the logs."""

import importlib
import logging
import threading

import numpy as np
import pytest

from backend import observability
from backend.jobs import gate as gate_module
from backend.lash_extraction import matting as matting_module


def _trimap(h: int, w: int) -> np.ndarray:
    trimap = np.zeros((h, w), np.uint8)
    trimap[h // 3 : 2 * h // 3] = 128
    trimap[h // 2 - 2 : h // 2 + 2, w // 2 - 8 : w // 2 + 8] = 255
    return trimap


@pytest.fixture(autouse=True)
def _restore_backend_logger():
    backend_logger = logging.getLogger("backend")
    handlers, level = list(backend_logger.handlers), backend_logger.level
    yield
    backend_logger.handlers[:] = handlers
    backend_logger.setLevel(level)


class TestLoggingSetup:
    """uvicorn leaves the root logger without a handler, so the app has to add its own."""

    def test_it_attaches_one_stream_handler_at_info(self, monkeypatch):
        backend_logger = logging.getLogger("backend")
        backend_logger.handlers.clear()
        monkeypatch.delenv("MATTE_LOG_LEVEL", raising=False)

        observability.configure_logging()
        observability.configure_logging()

        assert len(backend_logger.handlers) == 1
        assert backend_logger.level == logging.INFO

    def test_the_level_is_configurable(self, monkeypatch):
        monkeypatch.setenv("MATTE_LOG_LEVEL", "warning")

        observability.configure_logging()

        assert logging.getLogger("backend").level == logging.WARNING

    def test_an_unknown_level_still_leaves_the_matte_lines_visible(self, monkeypatch):
        monkeypatch.setenv("MATTE_LOG_LEVEL", "verbose")

        observability.configure_logging()

        assert logging.getLogger("backend").level == logging.INFO


class TestStartupLog:
    """Startup has to say which mode the process will use, so a Render deploy is checkable."""

    def test_it_logs_the_mode_budget_and_concurrency(self, monkeypatch, caplog):
        monkeypatch.setenv("MATTE_SOLVE_MODE", "tiled")
        monkeypatch.setenv("MATTE_MAX_SOLVE_PIXELS", "60000")
        monkeypatch.setenv("MATTE_MAX_WORKERS", "1")

        with caplog.at_level(logging.INFO):
            observability.log_matte_settings()

        assert "solve_mode=tiled" in caplog.text
        assert "max_solve_pixels=60000" in caplog.text
        assert "max_workers=1" in caplog.text

    def test_it_logs_the_face_detection_size_limit(self, monkeypatch, caplog):
        monkeypatch.setenv("MATTE_DETECT_MAX_SIDE", "1200")

        with caplog.at_level(logging.INFO):
            observability.log_matte_settings()

        assert "detect_max_side=1200" in caplog.text

    def test_a_misconfigured_detection_size_is_reported_at_startup(self, monkeypatch, caplog):
        # リクエスト時に 400 を返すだけだと、タイポなのか画像の問題なのか分からない
        monkeypatch.setenv("MATTE_DETECT_MAX_SIDE", "1600px")

        with caplog.at_level(logging.INFO):
            observability.log_matte_settings()

        assert "MATTE_DETECT_MAX_SIDE" in caplog.text
        assert caplog.records[-1].levelno >= logging.ERROR

    def test_it_says_the_defaults_are_defaults_when_nothing_is_configured(self, monkeypatch, caplog):
        for name in (
            "MATTE_SOLVE_MODE",
            "MATTE_MAX_SOLVE_PIXELS",
            "MATTE_MAX_WORKERS",
            "MATTE_DETECT_MAX_SIDE",
        ):
            monkeypatch.delenv(name, raising=False)

        with caplog.at_level(logging.INFO):
            observability.log_matte_settings()

        assert "solve_mode=full" in caplog.text
        assert "max_solve_pixels=none" in caplog.text
        assert "default" in caplog.text

    def test_a_misconfigured_mode_is_reported_instead_of_crashing_the_import(self, monkeypatch, caplog):
        monkeypatch.setenv("MATTE_SOLVE_MODE", "titled")

        with caplog.at_level(logging.INFO):
            observability.log_matte_settings()

        assert "titled" in caplog.text
        assert caplog.records[-1].levelno >= logging.ERROR

    def test_the_app_logs_its_settings_on_import(self, monkeypatch):
        logged: list[str] = []
        monkeypatch.setattr(observability, "log_matte_settings", lambda: logged.append("logged"))

        import backend.app

        importlib.reload(backend.app)

        assert logged == ["logged"]


class TestPerRunLog:
    """Every matting logs the mode it actually used, so an operator can confirm it per click."""

    def test_a_full_solve_logs_mode_size_and_duration(self, caplog):
        img = np.full((60, 90, 3), 40, np.uint8)
        img[28:32, 37:53] = 220

        with caplog.at_level(logging.INFO):
            matting_module.run_matting(img, _trimap(60, 90), mode="full")

        assert "solve_mode=full" in caplog.text
        assert "roi=90x60" in caplog.text
        assert "solves=1" in caplog.text
        assert "elapsed_ms=" in caplog.text

    def test_a_tiled_solve_logs_the_budget_and_how_many_solves_it_took(self, caplog):
        img = np.full((80, 120, 3), 40, np.uint8)
        img[38:42, 52:68] = 220

        with caplog.at_level(logging.INFO):
            matting_module.run_matting(img, _trimap(80, 120), mode="tiled", max_solve_pixels=1_500)

        assert "solve_mode=tiled" in caplog.text
        assert "max_solve_pixels=1500" in caplog.text
        assert "max_solve_px=" in caplog.text
        solves = int(caplog.text.split("solves=")[1].split()[0])
        assert solves > 1

    def test_a_degenerate_trimap_says_it_was_answered_without_a_solver(self, caplog):
        img = np.full((20, 20, 3), 40, np.uint8)

        with caplog.at_level(logging.INFO):
            matting_module.run_matting(img, np.zeros((20, 20), np.uint8), mode="full")

        assert "solves=0" in caplog.text


class TestGateLog:
    """A request waiting for the single slot is the thing to look for when matting feels slow."""

    def test_waiting_for_a_slot_is_logged(self, monkeypatch, caplog):
        monkeypatch.setattr(gate_module, "_state", gate_module._GateState())
        monkeypatch.setattr(gate_module, "release_memory", lambda: None)
        monkeypatch.setenv("MATTE_MAX_WORKERS", "1")
        holding, release, waited = threading.Event(), threading.Event(), threading.Event()

        def hold():
            with gate_module.matte_slot():
                holding.set()
                release.wait(timeout=5)

        def wait_for_slot():
            with gate_module.matte_slot():
                waited.set()

        holder = threading.Thread(target=hold)
        holder.start()
        holding.wait(timeout=5)
        with caplog.at_level(logging.INFO):
            waiter = threading.Thread(target=wait_for_slot)
            waiter.start()
            release.set()
            waiter.join(timeout=5)
        holder.join(timeout=5)

        assert waited.is_set()
        assert "waiting for a matting slot" in caplog.text

    def test_a_free_slot_is_not_announced(self, monkeypatch, caplog):
        monkeypatch.setattr(gate_module, "_state", gate_module._GateState())
        monkeypatch.setattr(gate_module, "release_memory", lambda: None)

        with caplog.at_level(logging.INFO), gate_module.matte_slot():
            pass

        assert "waiting for a matting slot" not in caplog.text
