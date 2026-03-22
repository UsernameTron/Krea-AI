"""Tests for main.py — CLI entry point commands.

Patching strategy for lazy imports:
  cmd_generate imports FluxKreaPipeline inside its body → patch "pipeline.FluxKreaPipeline"
  cmd_web imports create_ui inside its body → patch "app.create_ui"
  cmd_benchmark imports run_benchmark inside its body → patch "utils.benchmark.run_benchmark"
  cmd_info imports get_system_info inside its body → patch "utils.monitor.get_system_info"
  get_hf_token is imported at main module level → patch "main.get_hf_token"
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig

# Ensure gradio is mocked before any import that might trigger app.py loading
_mock_gradio = MagicMock()
_mock_gradio.Progress.return_value = MagicMock()
_mock_gradio.Blocks.return_value.__enter__ = MagicMock(return_value=MagicMock())
_mock_gradio.Blocks.return_value.__exit__ = MagicMock(return_value=False)
sys.modules.setdefault("gradio", _mock_gradio)


class TestCmdGenerate:
    def _make_args(self, **kwargs):
        args = MagicMock()
        args.prompt = kwargs.get("prompt", "a beautiful cat")
        args.width = kwargs.get("width", 1024)
        args.height = kwargs.get("height", 1024)
        args.steps = kwargs.get("steps", 28)
        args.guidance_scale = kwargs.get("guidance_scale", 4.5)
        args.seed = kwargs.get("seed", -1)
        return args

    def _make_pipeline(self, is_loaded=True):
        mock_pipeline = MagicMock()
        mock_pipeline.is_loaded = is_loaded
        mock_image = MagicMock()
        mock_pipeline.generate.return_value = (mock_image, {
            "generation_time": 5.0,
            "time_per_step": 0.5,
            "device": "mps",
            "seed": 42,
        })
        mock_pipeline.save_image.return_value = Path("/tmp/output.png")
        return mock_pipeline

    def test_cmd_generate_loads_pipeline(self):
        from main import cmd_generate

        config = FluxConfig()
        args = self._make_args()
        mock_pipeline = self._make_pipeline()

        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline):
            cmd_generate(config, args)

        mock_pipeline.load.assert_called_once()

    def test_cmd_generate_exits_if_not_loaded(self):
        from main import cmd_generate

        config = FluxConfig()
        args = self._make_args()
        mock_pipeline = self._make_pipeline(is_loaded=False)

        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline):
            with pytest.raises(SystemExit) as exc_info:
                cmd_generate(config, args)

        assert exc_info.value.code == 1

    def test_cmd_generate_negative_seed_uses_none(self):
        from main import cmd_generate

        config = FluxConfig()
        args = self._make_args(seed=-1)
        mock_pipeline = self._make_pipeline()

        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline):
            cmd_generate(config, args)

        call_kwargs = mock_pipeline.generate.call_args[1]
        assert call_kwargs["seed"] is None

    def test_cmd_generate_positive_seed_passes_value(self):
        from main import cmd_generate

        config = FluxConfig()
        args = self._make_args(seed=123)
        mock_pipeline = self._make_pipeline()
        mock_pipeline.generate.return_value = (MagicMock(), {
            "generation_time": 5.0,
            "time_per_step": 0.5,
            "device": "mps",
            "seed": 123,
        })

        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline):
            cmd_generate(config, args)

        call_kwargs = mock_pipeline.generate.call_args[1]
        assert call_kwargs["seed"] == 123

    def test_cmd_generate_unloads_pipeline(self):
        from main import cmd_generate

        config = FluxConfig()
        args = self._make_args()
        mock_pipeline = self._make_pipeline()

        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline):
            cmd_generate(config, args)

        mock_pipeline.unload.assert_called_once()


class TestCmdBenchmark:
    def test_cmd_benchmark_calls_run_benchmark(self):
        from main import cmd_benchmark

        config = FluxConfig()
        args = MagicMock()
        args.quick = False

        mock_results = {
            4: {"total_time": 2.0, "time_per_step": 0.5, "pixels_per_second": 500000},
            10: {"total_time": 5.0, "time_per_step": 0.5, "pixels_per_second": 200000},
        }

        with patch("utils.benchmark.run_benchmark", return_value=mock_results) as mock_bench:
            cmd_benchmark(config, args)

        mock_bench.assert_called_once_with(config=config, quick=False)

    def test_cmd_benchmark_quick_flag_passed(self):
        from main import cmd_benchmark

        config = FluxConfig()
        args = MagicMock()
        args.quick = True

        mock_results = {}

        with patch("utils.benchmark.run_benchmark", return_value=mock_results) as mock_bench:
            cmd_benchmark(config, args)

        mock_bench.assert_called_once_with(config=config, quick=True)

    def test_cmd_benchmark_handles_error_results(self, capsys):
        from main import cmd_benchmark

        config = FluxConfig()
        args = MagicMock()
        args.quick = False

        mock_results = {4: {"error": "pipeline failed"}}

        with patch("utils.benchmark.run_benchmark", return_value=mock_results):
            cmd_benchmark(config, args)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out


class TestCmdInfo:
    def test_cmd_info_prints_system_info(self, capsys):
        from main import cmd_info

        config = FluxConfig()
        args = MagicMock()

        mock_info = {
            "pytorch_version": "2.0.0",
            "mps_available": True,
            "cuda_available": False,
            "system_memory_gb": 32,
            "memory_used_percent": 50.0,
            "cpu_count": 8,
            "cpu_count_physical": 4,
        }

        with patch("utils.monitor.get_system_info", return_value=mock_info):
            with patch("main.get_hf_token", return_value="hf_test"):
                cmd_info(config, args)

        captured = capsys.readouterr()
        assert "2.0.0" in captured.out

    def test_cmd_info_shows_mps_available(self, capsys):
        from main import cmd_info

        config = FluxConfig()
        args = MagicMock()

        mock_info = {"mps_available": True}

        with patch("utils.monitor.get_system_info", return_value=mock_info):
            with patch("main.get_hf_token", return_value=None):
                cmd_info(config, args)

        captured = capsys.readouterr()
        assert "Available" in captured.out

    def test_cmd_info_shows_token_not_set_when_missing(self, capsys):
        from main import cmd_info

        config = FluxConfig()
        args = MagicMock()

        with patch("utils.monitor.get_system_info", return_value={}):
            with patch("main.get_hf_token", return_value=None):
                cmd_info(config, args)

        captured = capsys.readouterr()
        assert "NOT SET" in captured.out

    def test_cmd_info_shows_token_set_when_present(self, capsys):
        from main import cmd_info

        config = FluxConfig()
        args = MagicMock()

        with patch("utils.monitor.get_system_info", return_value={}):
            with patch("main.get_hf_token", return_value="hf_abc123"):
                cmd_info(config, args)

        captured = capsys.readouterr()
        assert "Set" in captured.out


class TestCmdWeb:
    def test_cmd_web_calls_create_ui(self):
        from main import cmd_web

        config = FluxConfig()
        args = MagicMock()
        args.port = None
        args.host = None
        args.share = False

        mock_ui = MagicMock()

        with patch("app.create_ui", return_value=mock_ui) as mock_create:
            cmd_web(config, args)

        mock_create.assert_called_once_with(config)

    def test_cmd_web_port_override(self):
        from main import cmd_web

        config = FluxConfig()
        args = MagicMock()
        args.port = 8080
        args.host = None
        args.share = False

        mock_ui = MagicMock()

        with patch("app.create_ui", return_value=mock_ui):
            cmd_web(config, args)

        assert config.web.port == 8080

    def test_cmd_web_share_override(self):
        from main import cmd_web

        config = FluxConfig()
        args = MagicMock()
        args.port = None
        args.host = None
        args.share = True

        mock_ui = MagicMock()

        with patch("app.create_ui", return_value=mock_ui):
            cmd_web(config, args)

        assert config.web.share is True


class TestMainFunction:
    def test_no_command_exits_zero(self):
        """main() with no subcommand prints help and exits 0."""
        from main import main

        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0

    def test_keyboard_interrupt_exits_130(self):
        from main import main

        with patch("sys.argv", ["main.py", "info"]):
            with patch("main.get_config", return_value=FluxConfig()):
                with patch("main.cmd_info", side_effect=KeyboardInterrupt):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
        assert exc_info.value.code == 130

    def test_exception_exits_1(self):
        from main import main

        with patch("sys.argv", ["main.py", "info"]):
            with patch("main.get_config", return_value=FluxConfig()):
                with patch("main.cmd_info", side_effect=RuntimeError("fatal")):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
        assert exc_info.value.code == 1

    def test_info_command_dispatches(self):
        from main import main

        with patch("sys.argv", ["main.py", "info"]):
            with patch("main.get_config", return_value=FluxConfig()):
                with patch("main.cmd_info") as mock_info:
                    main()
        mock_info.assert_called_once()

    def test_optimization_flag_applied(self):
        from main import main

        captured_config = []

        def capture_config(**kwargs):
            config = FluxConfig()
            config.optimization_level = kwargs.get("optimization_level", "standard")
            captured_config.append(config)
            return captured_config[-1]

        with patch("sys.argv", ["main.py", "--optimization", "none", "info"]):
            with patch("main.get_config", side_effect=capture_config):
                with patch("main.cmd_info"):
                    main()

        assert captured_config[0].optimization_level == "none"
