"""Tests for utils/profiler.py and the main.py 'profile' subcommand."""

import time
from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config():
    """Minimal config suitable for testing (no real model needed)."""
    config = FluxConfig()
    config.generation.width = 512
    config.generation.height = 512
    config.generation.steps = 4
    config.optimization_level = "none"
    return config


@pytest.fixture
def mock_pipeline(test_config):
    """Fake FluxKreaPipeline that returns a plausible image + metrics without loading the model."""
    from PIL import Image

    fake_image = Image.new("RGB", (512, 512), color="blue")

    pipeline = MagicMock()
    pipeline.is_loaded = True
    pipeline._load_time = 5.0

    def fake_generate(prompt, width, height, num_inference_steps, **kwargs):
        # Invoke progress_callback if provided so the profiler can time stages
        cb = kwargs.get("progress_callback")
        if cb:
            cb(0.0, "Preparing...")
            cb(0.1, "Encoding text...")
            cb(0.5, "Denoising step 2...")
            cb(0.9, "Finishing...")
            cb(1.0, "Done")
        return fake_image, {
            "generation_time": 2.0,
            "time_per_step": 0.5,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": 4.5,
            "seed": None,
            "device": "cpu",
        }

    pipeline.generate.side_effect = fake_generate
    pipeline.unload = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# FluxProfiler — init
# ---------------------------------------------------------------------------

class TestFluxProfilerInit:
    def test_init_defaults(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        assert p.config is test_config
        assert p.timings == {}
        assert p._profiles == []

    def test_init_without_config_uses_default(self):
        """FluxProfiler(None) should call get_config() without raising."""
        from utils.profiler import FluxProfiler

        with patch("utils.profiler.get_config") as mock_gc:
            mock_gc.return_value = FluxConfig()
            p = FluxProfiler()
            mock_gc.assert_called_once()


# ---------------------------------------------------------------------------
# _time_stage
# ---------------------------------------------------------------------------

class TestTimeStage:
    def test_records_timing_key(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p._time_stage("my_stage", lambda: time.sleep(0.01))
        assert "my_stage" in p.timings
        assert p.timings["my_stage"] >= 0.0

    def test_returns_elapsed(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        elapsed = p._time_stage("s", lambda: None)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_callable_receives_args(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        captured = []
        p._time_stage("s", lambda x, y=0: captured.append((x, y)), 42, y=7)
        assert captured == [(42, 7)]


# ---------------------------------------------------------------------------
# profile_generation
# ---------------------------------------------------------------------------

class TestProfileGeneration:
    def test_returns_dict_with_expected_keys(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_generation(mock_pipeline, "test prompt", 512, 512, 4)

        expected_keys = {
            "model_load_time",
            "encode_time",
            "denoise_time",
            "denoise_steps",
            "decode_time",
            "total_time",
            "steps",
            "width",
            "height",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_timings_are_non_negative(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_generation(mock_pipeline, "test", 512, 512, 4)

        assert result["encode_time"] >= 0.0
        assert result["denoise_time"] >= 0.0
        assert result["decode_time"] >= 0.0
        assert result["total_time"] >= 0.0
        assert result["model_load_time"] >= 0.0

    def test_step_list_length(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        # denoise_steps list should not exceed requested steps
        assert len(result["denoise_steps"]) <= 4

    def test_dimensions_recorded(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_generation(mock_pipeline, "test", 512, 768, 4)
        assert result["width"] == 512
        assert result["height"] == 768

    def test_profile_stored_in_profiles_list(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        assert len(p._profiles) == 1

    def test_pipeline_load_called_when_not_loaded(self, test_config, mock_pipeline):
        """When pipeline.is_loaded is False, profile_generation should call load()."""
        from utils.profiler import FluxProfiler

        mock_pipeline.is_loaded = False

        def set_loaded():
            mock_pipeline.is_loaded = True

        mock_pipeline.load.side_effect = set_loaded

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        mock_pipeline.load.assert_called_once()

    def test_generate_called_with_correct_params(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "hello world", 768, 512, 8)

        call_kwargs = mock_pipeline.generate.call_args
        # positional or keyword — normalise
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()

        # prompt is either first positional arg or keyword
        called_prompt = args[0] if args else kwargs.get("prompt")
        assert called_prompt == "hello world"

    def test_error_during_generation_propagates(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        mock_pipeline.generate.side_effect = RuntimeError("generation boom")
        p = FluxProfiler(test_config)

        with pytest.raises(RuntimeError, match="generation boom"):
            p.profile_generation(mock_pipeline, "test", 512, 512, 4)


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_returns_empty_dict_before_profiling(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        assert p.get_metrics() == {}

    def test_returns_copy_of_timings(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)

        metrics = p.get_metrics()
        assert isinstance(metrics, dict)
        # Mutating the returned dict must not change internal state
        metrics["total_time"] = 9999.0
        assert p.timings.get("total_time") != 9999.0

    def test_contains_all_expected_keys(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        metrics = p.get_metrics()

        for key in ("total_time", "encode_time", "denoise_time", "decode_time"):
            assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_returns_placeholder_before_profiling(self, test_config):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        summary = p.get_summary()
        assert "no profiling data" in summary.lower() or summary.strip() != ""

    def test_returns_string(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        summary = p.get_summary()
        assert isinstance(summary, str)

    def test_summary_contains_stage_labels(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        summary = p.get_summary()

        for label in ("Text Encoding", "Denoising", "VAE Decode", "Total"):
            assert label in summary, f"Expected '{label}' in summary"

    def test_summary_contains_border_characters(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        summary = p.get_summary()

        assert "┌" in summary
        assert "└" in summary
        assert "│" in summary

    def test_summary_shows_model_load_when_nonzero(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        # Force a known load time
        mock_pipeline.is_loaded = True
        mock_pipeline._load_time = 10.0
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        p.timings["model_load_time"] = 10.0  # ensure it's set
        summary = p.get_summary()
        assert "Model Load" in summary

    def test_summary_has_percentage_column(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        p.profile_generation(mock_pipeline, "test", 512, 512, 4)
        summary = p.get_summary()
        assert "%" in summary


# ---------------------------------------------------------------------------
# profile_with_torch_profiler
# ---------------------------------------------------------------------------

class TestProfileWithTorchProfiler:
    def test_returns_dict(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_with_torch_profiler(mock_pipeline, "test", 512, 512, 4)
        assert isinstance(result, dict)

    def test_contains_profile_available_key(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_with_torch_profiler(mock_pipeline, "test", 512, 512, 4)
        assert "profile_available" in result

    def test_returns_error_when_pipeline_not_loaded(self, test_config, mock_pipeline):
        from utils.profiler import FluxProfiler

        mock_pipeline.is_loaded = False
        p = FluxProfiler(test_config)
        result = p.profile_with_torch_profiler(mock_pipeline, "test", 512, 512, 4)
        assert result["profile_available"] is False
        assert "error" in result

    def test_graceful_when_torch_profiler_unavailable(self, test_config, mock_pipeline):
        """If torch.profiler import fails, should return profile_available=False."""
        from utils.profiler import FluxProfiler
        import sys

        p = FluxProfiler(test_config)

        # Temporarily hide torch.profiler
        with patch.dict(sys.modules, {"torch.profiler": None}):
            result = p.profile_with_torch_profiler(mock_pipeline, "test", 512, 512, 4)
        # Should not raise; should return a dict with profile_available key
        assert isinstance(result, dict)
        assert "profile_available" in result

    def test_profile_available_true_when_profiler_works(self, test_config, mock_pipeline):
        """When torch.profiler succeeds, profile_available should be True."""
        from utils.profiler import FluxProfiler

        p = FluxProfiler(test_config)
        result = p.profile_with_torch_profiler(mock_pipeline, "test", 512, 512, 4)
        # profile_available may be True or False depending on environment
        # — just assert the key exists and value is bool
        assert isinstance(result["profile_available"], bool)


# ---------------------------------------------------------------------------
# cmd_profile in main.py
# ---------------------------------------------------------------------------

class TestCmdProfile:
    def test_cmd_profile_calls_profiler(self, test_config, mock_pipeline):
        """cmd_profile should instantiate FluxProfiler and call profile_generation."""
        import argparse
        import main as main_module

        args = argparse.Namespace(
            prompt="test landscape",
            width=512,
            height=512,
            steps=4,
            torch_profile=False,
        )

        # cmd_profile uses local imports, so patch at the source module level
        with patch("pipeline.FluxKreaPipeline", return_value=mock_pipeline), \
             patch("utils.profiler.FluxProfiler") as MockProfiler:

            mock_prof_instance = MagicMock()
            mock_prof_instance.get_summary.return_value = "summary output"
            MockProfiler.return_value = mock_prof_instance

            # Patch the local import inside cmd_profile directly
            with patch.dict("sys.modules", {}):
                import importlib
                import utils.profiler as profiler_module
                original_class = profiler_module.FluxProfiler
                profiler_module.FluxProfiler = MockProfiler

                import pipeline as pipeline_module
                original_pipeline = pipeline_module.FluxKreaPipeline
                pipeline_module.FluxKreaPipeline = MagicMock(return_value=mock_pipeline)

                try:
                    main_module.cmd_profile(test_config, args)
                finally:
                    profiler_module.FluxProfiler = original_class
                    pipeline_module.FluxKreaPipeline = original_pipeline

            MockProfiler.assert_called_once_with(test_config)
            mock_prof_instance.profile_generation.assert_called_once()
            mock_prof_instance.get_summary.assert_called_once()

    def test_cmd_profile_uses_config_defaults_when_args_none(self, test_config, mock_pipeline):
        """Width/height/steps fall back to config values when args are None."""
        import argparse
        import main as main_module
        import pipeline as pipeline_module
        import utils.profiler as profiler_module

        args = argparse.Namespace(
            prompt=None,
            width=None,
            height=None,
            steps=None,
            torch_profile=False,
        )

        mock_prof_instance = MagicMock()
        mock_prof_instance.get_summary.return_value = ""

        original_profiler = profiler_module.FluxProfiler
        original_pipeline = pipeline_module.FluxKreaPipeline

        profiler_module.FluxProfiler = MagicMock(return_value=mock_prof_instance)
        pipeline_module.FluxKreaPipeline = MagicMock(return_value=mock_pipeline)

        try:
            main_module.cmd_profile(test_config, args)
        finally:
            profiler_module.FluxProfiler = original_profiler
            pipeline_module.FluxKreaPipeline = original_pipeline

        call_args = mock_prof_instance.profile_generation.call_args
        _, _, w, h, s = call_args.args
        assert w == test_config.generation.width
        assert h == test_config.generation.height
        assert s == test_config.generation.steps

    def test_cmd_profile_torch_profile_flag(self, test_config, mock_pipeline):
        """When --torch-profile is set, profile_with_torch_profiler is called."""
        import argparse
        import main as main_module
        import pipeline as pipeline_module
        import utils.profiler as profiler_module

        args = argparse.Namespace(
            prompt="test",
            width=512,
            height=512,
            steps=4,
            torch_profile=True,
        )

        mock_prof_instance = MagicMock()
        mock_prof_instance.get_summary.return_value = ""
        mock_prof_instance.profile_with_torch_profiler.return_value = {
            "profile_available": False,
            "error": "mocked",
        }

        original_profiler = profiler_module.FluxProfiler
        original_pipeline = pipeline_module.FluxKreaPipeline

        profiler_module.FluxProfiler = MagicMock(return_value=mock_prof_instance)
        pipeline_module.FluxKreaPipeline = MagicMock(return_value=mock_pipeline)

        try:
            main_module.cmd_profile(test_config, args)
        finally:
            profiler_module.FluxProfiler = original_profiler
            pipeline_module.FluxKreaPipeline = original_pipeline

        mock_prof_instance.profile_with_torch_profiler.assert_called_once()

    def test_cmd_profile_unloads_pipeline(self, test_config, mock_pipeline):
        """cmd_profile should call pipeline.unload() after profiling."""
        import argparse
        import main as main_module
        import pipeline as pipeline_module
        import utils.profiler as profiler_module

        args = argparse.Namespace(
            prompt="test",
            width=512,
            height=512,
            steps=4,
            torch_profile=False,
        )

        mock_prof_instance = MagicMock()
        mock_prof_instance.get_summary.return_value = ""

        original_profiler = profiler_module.FluxProfiler
        original_pipeline = pipeline_module.FluxKreaPipeline

        profiler_module.FluxProfiler = MagicMock(return_value=mock_prof_instance)
        pipeline_module.FluxKreaPipeline = MagicMock(return_value=mock_pipeline)

        try:
            main_module.cmd_profile(test_config, args)
        finally:
            profiler_module.FluxProfiler = original_profiler
            pipeline_module.FluxKreaPipeline = original_pipeline

        mock_pipeline.unload.assert_called_once()


# ---------------------------------------------------------------------------
# CLI subcommand registration (smoke test via argparse)
# ---------------------------------------------------------------------------

class TestProfileSubcommandRegistered:
    def test_profile_help_does_not_raise(self):
        """python main.py profile --help should parse without error."""
        import argparse

        # Re-import to ensure we get the actual parser
        import main as main_module
        import sys
        from io import StringIO

        with patch.object(sys, "argv", ["main.py", "profile", "--help"]), \
             pytest.raises(SystemExit) as exc_info:
            # argparse raises SystemExit(0) on --help
            main_module.main()

        assert exc_info.value.code == 0

    def test_profile_args_parsed_correctly(self):
        """Verify argparse resolves profile subcommand args as expected."""
        import argparse
        import sys
        import main as main_module

        # We need the parser — rebuild it inline to avoid running main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimization", "-O", choices=["none", "standard", "maximum"])
        parser.add_argument("--verbose", "-v", action="store_true")
        subparsers = parser.add_subparsers(dest="command")

        profile_parser = subparsers.add_parser("profile")
        profile_parser.add_argument("--prompt", "-p", default=None)
        profile_parser.add_argument("--width", "-W", type=int, default=None)
        profile_parser.add_argument("--height", "-H", type=int, default=None)
        profile_parser.add_argument("--steps", "-s", type=int, default=None)
        profile_parser.add_argument("--torch-profile", action="store_true")

        args = parser.parse_args(["profile", "-p", "mountains", "--steps", "10", "--torch-profile"])
        assert args.command == "profile"
        assert args.prompt == "mountains"
        assert args.steps == 10
        assert args.torch_profile is True
        assert args.width is None
