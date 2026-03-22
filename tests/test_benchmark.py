"""Tests for utils/benchmark.py — run_benchmark()."""

from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig


class TestRunBenchmark:
    def _mock_pipeline(self, steps_to_fail=None):
        """Create a mock FluxKreaPipeline that succeeds or fails per step count."""
        mock_pipeline = MagicMock()
        mock_image = MagicMock()

        def generate_side_effect(**kwargs):
            steps = kwargs.get("num_inference_steps", 4)
            if steps_to_fail and steps in steps_to_fail:
                raise RuntimeError(f"failed at {steps} steps")
            return mock_image, {
                "generation_time": steps * 0.5,
                "time_per_step": 0.5,
                "device": "mps",
                "seed": 42,
            }

        mock_pipeline.generate.side_effect = generate_side_effect
        return mock_pipeline

    def test_returns_dict(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=self._mock_pipeline()):
            result = run_benchmark(config=config, steps_list=[4])
        assert isinstance(result, dict)

    def test_default_steps_list_used_when_none(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        mock_pipeline = self._mock_pipeline()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            result = run_benchmark(config=config)
        # Default steps_list is [4, 10, 20]
        assert 4 in result
        assert 10 in result
        assert 20 in result

    def test_quick_mode_uses_smaller_steps_list(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        mock_pipeline = self._mock_pipeline()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            result = run_benchmark(config=config, quick=True)
        # Quick mode uses [4, 10] only
        assert 4 in result
        assert 10 in result
        assert 20 not in result

    def test_quick_mode_uses_512x512(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        config.generation.width = 1024
        config.generation.height = 1024

        call_kwargs_list = []
        mock_image = MagicMock()

        def capture_generate(**kwargs):
            call_kwargs_list.append(kwargs)
            return mock_image, {"generation_time": 2.0, "time_per_step": 0.5}

        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = capture_generate

        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            run_benchmark(config=config, steps_list=[4], quick=True)

        assert call_kwargs_list[0]["width"] == 512
        assert call_kwargs_list[0]["height"] == 512

    def test_non_quick_uses_config_dimensions(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        config.generation.width = 768
        config.generation.height = 768

        call_kwargs_list = []
        mock_image = MagicMock()

        def capture_generate(**kwargs):
            call_kwargs_list.append(kwargs)
            return mock_image, {"generation_time": 2.0, "time_per_step": 0.5}

        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = capture_generate

        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            run_benchmark(config=config, steps_list=[4], quick=False)

        assert call_kwargs_list[0]["width"] == 768
        assert call_kwargs_list[0]["height"] == 768

    def test_results_contain_total_time(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=self._mock_pipeline()):
            result = run_benchmark(config=config, steps_list=[4])
        assert "total_time" in result[4]

    def test_results_contain_time_per_step(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=self._mock_pipeline()):
            result = run_benchmark(config=config, steps_list=[4])
        assert "time_per_step" in result[4]

    def test_results_contain_pixels_per_second(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=self._mock_pipeline()):
            result = run_benchmark(config=config, steps_list=[4])
        assert "pixels_per_second" in result[4]

    def test_failed_step_stored_as_error(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=self._mock_pipeline(steps_to_fail=[10])):
            result = run_benchmark(config=config, steps_list=[4, 10])
        assert "error" in result[10]
        assert "total_time" in result[4]  # step 4 should succeed

    def test_unloads_pipeline_after_run(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        mock_pipeline = self._mock_pipeline()
        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            run_benchmark(config=config, steps_list=[4])
        mock_pipeline.unload.assert_called_once()

    def test_uses_get_config_when_no_config_provided(self):
        from utils.benchmark import run_benchmark

        mock_pipeline = self._mock_pipeline()
        mock_config = FluxConfig()

        with patch("utils.benchmark.get_config", return_value=mock_config) as mock_get_config:
            with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
                run_benchmark()

        mock_get_config.assert_called_once()

    def test_pixels_per_second_calculation(self):
        from utils.benchmark import run_benchmark

        config = FluxConfig()
        config.generation.width = 512
        config.generation.height = 512

        mock_image = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = (mock_image, {
            "generation_time": 4.0,
            "time_per_step": 1.0,
        })

        with patch("utils.benchmark.FluxKreaPipeline", return_value=mock_pipeline):
            result = run_benchmark(config=config, steps_list=[4], quick=False)

        # 512 * 512 / 4.0 = 65536.0
        expected = 512 * 512 / 4.0
        assert abs(result[4]["pixels_per_second"] - expected) < 1.0
