"""Tests for app.py — FluxWebApp and create_ui.

gradio is mocked at the module level (before any import of app) because
gr.Progress() is evaluated at import time as a default parameter value.
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# ------------------------------------------------------------------
# Inject a full gradio mock into sys.modules BEFORE importing app.
# gr.Progress() is used as a default param value, so it's evaluated
# at import time. The mock must be in place before app is imported.
# ------------------------------------------------------------------
_mock_gradio = MagicMock()
_mock_gradio.Progress.return_value = MagicMock()
# Blocks/Row/Column/Accordion are used as context managers in create_ui
_mock_gradio.Blocks.return_value.__enter__ = MagicMock(return_value=MagicMock())
_mock_gradio.Blocks.return_value.__exit__ = MagicMock(return_value=False)
_mock_gradio.Row.return_value.__enter__ = MagicMock(return_value=MagicMock())
_mock_gradio.Row.return_value.__exit__ = MagicMock(return_value=False)
_mock_gradio.Column.return_value.__enter__ = MagicMock(return_value=MagicMock())
_mock_gradio.Column.return_value.__exit__ = MagicMock(return_value=False)
_mock_gradio.Accordion.return_value.__enter__ = MagicMock(return_value=MagicMock())
_mock_gradio.Accordion.return_value.__exit__ = MagicMock(return_value=False)
sys.modules["gradio"] = _mock_gradio

# Now it's safe to import app
from config import FluxConfig  # noqa: E402
from app import FluxWebApp  # noqa: E402


# Helper: access the module-level _debug_log via the app module
import app as app_module  # noqa: E402


def _make_app():
    """Create a FluxWebApp with a fully mocked pipeline."""
    config = FluxConfig()
    with patch("app.FluxKreaPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.is_loaded = False
        mock_pipeline_cls.return_value = mock_pipeline
        web_app = FluxWebApp(config)
        web_app.pipeline = mock_pipeline
    return web_app


class TestFluxWebAppInit:
    def test_init_creates_pipeline(self):
        config = FluxConfig()
        with patch("app.FluxKreaPipeline") as mock_cls:
            mock_cls.return_value = MagicMock()
            web_app = FluxWebApp(config)
        assert web_app.config is config
        mock_cls.assert_called_once_with(config)

    def test_init_sets_load_event(self):
        config = FluxConfig()
        with patch("app.FluxKreaPipeline"):
            web_app = FluxWebApp(config)
        assert isinstance(web_app._load_event, threading.Event)
        assert not web_app._load_event.is_set()

    def test_init_empty_load_error(self):
        config = FluxConfig()
        with patch("app.FluxKreaPipeline"):
            web_app = FluxWebApp(config)
        assert web_app._load_error == ""

    def test_init_empty_session_gallery(self):
        config = FluxConfig()
        with patch("app.FluxKreaPipeline"):
            web_app = FluxWebApp(config)
        assert web_app._session_gallery == []


class TestBackgroundLoad:
    def test_background_load_success_sets_event(self):
        web_app = _make_app()
        web_app.pipeline.load = MagicMock()
        web_app._background_load()
        assert web_app._load_event.is_set()

    def test_background_load_failure_sets_event(self):
        web_app = _make_app()
        web_app.pipeline.load = MagicMock(side_effect=RuntimeError("load failed"))
        web_app._background_load()
        assert web_app._load_event.is_set()

    def test_background_load_failure_stores_error(self):
        web_app = _make_app()
        web_app.pipeline.load = MagicMock(side_effect=RuntimeError("load failed"))
        web_app._background_load()
        assert "load failed" in web_app._load_error

    def test_background_load_success_clears_error(self):
        web_app = _make_app()
        web_app.pipeline.load = MagicMock()
        web_app._load_error = ""
        web_app._background_load()
        assert web_app._load_error == ""


class TestStartLoading:
    def test_start_loading_spawns_thread(self):
        web_app = _make_app()
        with patch.object(web_app, "_background_load"):
            web_app.start_loading()
        # The thread should be spawned (test completes without hang)

    def test_start_loading_daemon_thread(self):
        threads_started = []
        original_thread = threading.Thread

        class CapturingThread(original_thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                threads_started.append(self)

        web_app = _make_app()
        web_app._load_event.set()  # Prevent actual loading
        with patch("app.threading.Thread", CapturingThread):
            with patch.object(web_app, "_background_load", return_value=None):
                web_app.start_loading()

        assert len(threads_started) == 1
        assert threads_started[0].daemon is True


class TestGenerate:
    def _ready_app(self):
        """App with pipeline loaded and ready."""
        web_app = _make_app()
        web_app._load_event.set()
        web_app.pipeline.is_loaded = True
        mock_image = MagicMock()
        metrics = {
            "generation_time": 5.0,
            "time_per_step": 0.5,
            "device": "mps",
            "seed": 42,
        }
        web_app.pipeline.generate = MagicMock(return_value=(mock_image, metrics))
        web_app.pipeline.save_image = MagicMock(return_value="/tmp/test.png")
        return web_app

    def test_empty_prompt_returns_none(self):
        web_app = self._ready_app()
        result = web_app.generate("   ", 1024, 1024, 28, 4.5, -1, False, 5)
        assert result[0] is None
        assert "prompt" in result[1].lower()

    def test_load_error_returns_none(self):
        web_app = _make_app()
        web_app._load_event.set()
        web_app._load_error = "something went wrong"
        result = web_app.generate("test prompt", 1024, 1024, 28, 4.5, -1, False, 5)
        assert result[0] is None

    def test_pipeline_not_loaded_returns_none(self):
        web_app = _make_app()
        web_app._load_event.set()
        web_app.pipeline.is_loaded = False
        result = web_app.generate("test prompt", 1024, 1024, 28, 4.5, -1, False, 5)
        assert result[0] is None

    def test_successful_generation_returns_image(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            result = web_app.generate("a beautiful sunset", 1024, 1024, 28, 4.5, 42, False, 5)
        assert result[0] is not None

    def test_successful_generation_returns_info_string(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            result = web_app.generate("a beautiful sunset", 1024, 1024, 28, 4.5, 42, False, 5)
        assert isinstance(result[1], str)
        assert "1024x1024" in result[1]

    def test_successful_generation_adds_to_gallery(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            web_app.generate("a beautiful sunset", 1024, 1024, 28, 4.5, 42, False, 5)
        assert len(web_app._session_gallery) == 1

    def test_negative_seed_passes_none_to_pipeline(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            web_app.generate("test", 1024, 1024, 28, 4.5, -1, False, 5)
        call_kwargs = web_app.pipeline.generate.call_args[1]
        assert call_kwargs["seed"] is None

    def test_positive_seed_passes_value_to_pipeline(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            web_app.generate("test", 1024, 1024, 28, 4.5, 99, False, 5)
        call_kwargs = web_app.pipeline.generate.call_args[1]
        assert call_kwargs["seed"] == 99

    def test_safety_mode_true_sets_config(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            web_app.generate("test", 1024, 1024, 28, 4.5, -1, True, 5)
        assert web_app.config.safety_mode.enabled is True

    def test_safety_mode_false_clears_config(self):
        web_app = self._ready_app()
        web_app.config.safety_mode.enabled = True
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            web_app.generate("test", 1024, 1024, 28, 4.5, -1, False, 5)
        assert web_app.config.safety_mode.enabled is False

    def test_concurrency_lock_prevents_parallel_calls(self):
        web_app = self._ready_app()
        # Hold the lock
        web_app._lock.acquire()
        try:
            result = web_app.generate("test", 1024, 1024, 28, 4.5, -1, False, 5)
        finally:
            web_app._lock.release()
        assert result[0] is None
        assert "in progress" in result[1].lower()

    def test_generation_exception_returns_error(self):
        web_app = self._ready_app()
        web_app.pipeline.generate = MagicMock(side_effect=RuntimeError("generation failed"))
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            result = web_app.generate("test", 1024, 1024, 28, 4.5, -1, False, 5)
        assert result[0] is None

    def test_returns_debug_text_as_third_element(self):
        web_app = self._ready_app()
        with patch("app.signal.signal"), patch("app.signal.alarm"):
            result = web_app.generate("a cat", 1024, 1024, 28, 4.5, 42, False, 5)
        assert isinstance(result[2], str)


class TestFormatError:
    def test_generic_error(self):
        web_app = _make_app()
        msg = web_app._format_error("something went wrong")
        assert "something went wrong" in msg

    def test_gated_error_includes_hf_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("Repository is gated")
        assert "huggingface.co" in msg.lower() or "HF_TOKEN" in msg

    def test_401_error_includes_auth_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("401 Unauthorized")
        assert "token" in msg.lower() or "Authentication" in msg

    def test_403_error_includes_auth_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("403 Forbidden")
        assert "token" in msg.lower() or "Authentication" in msg

    def test_memory_error_includes_memory_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("out of memory: CUDA OOM")
        assert "memory" in msg.lower() or "resolution" in msg.lower()

    def test_oom_error_includes_memory_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("OOM error occurred")
        assert "memory" in msg.lower() or "resolution" in msg.lower()

    def test_timeout_error_includes_timeout_solution(self):
        web_app = _make_app()
        msg = web_app._format_error("timeout exceeded")
        assert "timeout" in msg.lower() or "Timeout" in msg


class TestGetDebugText:
    def test_returns_string(self):
        web_app = _make_app()
        result = web_app._get_debug_text()
        assert isinstance(result, str)

    def test_no_messages_returns_placeholder(self):
        web_app = _make_app()
        # Clear module-level debug log
        app_module._debug_log.clear()
        result = web_app._get_debug_text()
        assert result == "No debug messages yet."

    def test_messages_returned_when_present(self):
        web_app = _make_app()
        app_module._debug_log.clear()
        app_module._debug_log.append("test message")
        result = web_app._get_debug_text()
        assert "test message" in result


class TestGetSystemInfoText:
    def test_returns_string(self):
        web_app = _make_app()
        mock_info = {
            "pytorch_version": "2.0.0",
            "device": "mps",
            "mps_available": True,
            "system_memory_gb": 32,
            "hf_token_set": True,
            "optimization_level": "standard",
            "is_loaded": False,
        }
        web_app.pipeline.get_system_info = MagicMock(return_value=mock_info)
        result = web_app._get_system_info_text()
        assert isinstance(result, str)

    def test_includes_pytorch_version(self):
        web_app = _make_app()
        web_app.pipeline.get_system_info = MagicMock(return_value={
            "pytorch_version": "2.1.0",
            "device": "cpu",
            "mps_available": False,
        })
        result = web_app._get_system_info_text()
        assert "2.1.0" in result

    def test_shows_mps_allocated_when_present(self):
        web_app = _make_app()
        web_app.pipeline.get_system_info = MagicMock(return_value={
            "mps_available": True,
            "mps_allocated_mb": 512,
        })
        result = web_app._get_system_info_text()
        assert "512" in result


class TestCreateUI:
    def test_create_ui_returns_blocks(self):
        """create_ui should construct a gr.Blocks UI without error."""
        from app import create_ui
        config = FluxConfig()
        with patch("app.FluxKreaPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.get_system_info = MagicMock(return_value={})
            mock_cls.return_value = mock_pipeline
            # Should not raise
            ui = create_ui(config)
        assert ui is not None
