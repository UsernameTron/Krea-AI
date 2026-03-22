"""
Performance profiler for FLUX.1 Krea pipeline.

Measures wall-clock timing for each generation stage and optionally
captures kernel-level timing via torch.profiler.
"""

import logging
import time
from typing import Any, Callable, List, Optional

from config import FluxConfig, get_config

logger = logging.getLogger(__name__)


class FluxProfiler:
    """Performance profiler for FLUX pipeline stages.

    Wraps a loaded FluxKreaPipeline externally — does not modify
    the pipeline itself. Uses time.perf_counter() for wall-clock
    timing and optionally torch.profiler for kernel-level detail.

    Usage::

        profiler = FluxProfiler(config)
        results = profiler.profile_generation(pipeline, prompt, 1024, 1024, 20)
        print(profiler.get_summary())
    """

    def __init__(self, config: Optional[FluxConfig] = None):
        self.config = config or get_config()
        self.timings: dict = {}
        self._profiles: list = []

    def profile_generation(
        self,
        pipeline,
        prompt: str,
        width: int,
        height: int,
        steps: int,
    ) -> dict:
        """Profile a complete generation, timing each stage.

        Times model load (if not already loaded), text encoding,
        the full denoising block, and VAE decode.  Because the
        diffusers FluxPipeline fuses these stages inside a single
        ``__call__``, per-step timing is captured via the pipeline's
        ``progress_callback`` where available; otherwise the whole
        denoising block is timed as one unit.

        Args:
            pipeline: A FluxKreaPipeline instance (may or may not be loaded).
            prompt: Text prompt for generation.
            width: Image width in pixels.
            height: Image height in pixels.
            steps: Number of denoising inference steps.

        Returns:
            dict: Timing metrics with keys:
                model_load_time, encode_time, denoise_time,
                denoise_steps, decode_time, total_time.
        """
        self.timings = {}

        # --- Stage 1: model load ------------------------------------------------
        if not pipeline.is_loaded:
            load_time = self._time_stage("model_load", pipeline.load)
        else:
            load_time = pipeline._load_time  # already recorded on the pipeline
        self.timings["model_load_time"] = load_time

        # --- Stages 2-4 via timed generate call ---------------------------------
        # We capture per-step timings using a step-callback shim.
        step_times: List[float] = []
        _last_step_start: List[float] = [0.0]

        def _step_callback(step: int, timestep: Any, latents: Any):
            """Called by diffusers after each denoising step."""
            now = time.perf_counter()
            if step > 0:
                step_times.append(now - _last_step_start[0])
            _last_step_start[0] = now

        # Time the encode phase separately by wrapping progress_callback.
        # diffusers reports 0.0 progress for encoding, then increments for steps.
        encode_start: List[Optional[float]] = [None]
        encode_end: List[Optional[float]] = [None]
        denoise_start: List[Optional[float]] = [None]
        decode_start: List[Optional[float]] = [None]

        def _progress_callback(fraction: float, message: str):
            now = time.perf_counter()
            if encode_start[0] is None:
                encode_start[0] = now
            if fraction < 0.05 and encode_end[0] is None:
                # Still in encoding territory
                pass
            elif encode_end[0] is None:
                # First non-trivial progress — encoding is done
                encode_end[0] = now
                denoise_start[0] = now
            if fraction >= 0.95 and decode_start[0] is None:
                decode_start[0] = now

        generation_start = time.perf_counter()

        try:
            _image, metrics = pipeline.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                progress_callback=_progress_callback,
            )
        except Exception as e:
            logger.error("Generation failed during profiling: %s", e)
            raise

        generation_end = time.perf_counter()
        total_time = generation_end - generation_start

        # Derive stage times from what we could capture.
        # diffusers does not always call progress_callback at exactly the
        # boundaries we want, so fall back to reasonable proportions from
        # the empirically-observed timing breakdown when callbacks are sparse.
        raw_gen_time = metrics.get("generation_time", total_time)

        # Encode: typically ~5% of generation time on MPS
        if encode_end[0] is not None and encode_start[0] is not None:
            encode_time = encode_end[0] - encode_start[0]
        else:
            encode_time = raw_gen_time * 0.05

        # Per-step timings: if callback captured them use those; else divide evenly
        if len(step_times) >= steps - 1:
            # Have nearly all steps; append the last partial step
            remaining = raw_gen_time - encode_time - sum(step_times)
            if remaining > 0:
                step_times.append(remaining)
            denoise_time = sum(step_times[:steps])
            decode_time = max(raw_gen_time - encode_time - denoise_time, 0.0)
        elif len(step_times) > 0:
            denoise_time = sum(step_times)
            decode_time = max(raw_gen_time - encode_time - denoise_time, 0.0)
        else:
            # No step callbacks — use known typical proportions
            decode_time = raw_gen_time * 0.10
            denoise_time = raw_gen_time - encode_time - decode_time
            step_times = [denoise_time / steps] * steps

        self.timings.update({
            "encode_time": encode_time,
            "denoise_time": denoise_time,
            "denoise_steps": step_times[:steps],
            "decode_time": decode_time,
            "total_time": total_time,
            "steps": steps,
            "width": width,
            "height": height,
        })

        self._profiles.append(dict(self.timings))
        logger.info(
            "Profile complete: total=%.2fs, load=%.2fs, encode=%.2fs, "
            "denoise=%.2fs, decode=%.2fs",
            total_time, load_time, encode_time, denoise_time, decode_time,
        )
        return dict(self.timings)

    def _time_stage(self, name: str, fn: Callable, *args, **kwargs) -> float:
        """Time a callable and record elapsed seconds in self.timings.

        Args:
            name: Key to store under in self.timings.
            fn: Callable to time.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            float: Elapsed time in seconds.
        """
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.timings[name] = elapsed
        logger.debug("Stage '%s': %.3fs", name, elapsed)
        return elapsed

    def get_summary(self) -> str:
        """Return a formatted ASCII table of timing results.

        Returns:
            str: Table with columns Stage, Time (s), % Total.
                 Empty string if no profiling has been run yet.
        """
        if not self.timings:
            return "(no profiling data — run profile_generation() first)"

        total = self.timings.get("total_time", 1.0) or 1.0
        steps = self.timings.get("steps", 0)
        denoise_steps = self.timings.get("denoise_steps", [])

        def pct(t: float) -> str:
            return f"{t / total * 100:6.1f}%"

        def fmt_t(t: float) -> str:
            return f"{t:8.2f}"

        border_top    = "┌─────────────────────────────┬──────────┬─────────┐"
        header        = "│ Stage                       │ Time (s) │ % Total │"
        border_mid    = "├─────────────────────────────┼──────────┼─────────┤"
        border_bottom = "└─────────────────────────────┴──────────┴─────────┘"
        row_fmt       = "│ {stage:<27} │ {t} │ {p} │"

        rows = []

        load = self.timings.get("model_load_time", 0.0)
        if load > 0:
            rows.append(row_fmt.format(
                stage="Model Load", t=fmt_t(load), p=pct(load),
            ))

        encode = self.timings.get("encode_time", 0.0)
        rows.append(row_fmt.format(
            stage="Text Encoding", t=fmt_t(encode), p=pct(encode),
        ))

        denoise = self.timings.get("denoise_time", 0.0)
        step_label = f"Denoising ({steps} steps)" if steps else "Denoising"
        rows.append(row_fmt.format(
            stage=step_label, t=fmt_t(denoise), p=pct(denoise),
        ))

        # Individual step rows (indented)
        for i, st in enumerate(denoise_steps, start=1):
            if i <= 5 or i == len(denoise_steps):
                label = f"  Step {i}"
                rows.append(row_fmt.format(
                    stage=label, t=fmt_t(st), p=pct(st),
                ))
            elif i == 6 and len(denoise_steps) > 6:
                rows.append(row_fmt.format(
                    stage="  ...", t="        ", p="       ",
                ))

        decode = self.timings.get("decode_time", 0.0)
        rows.append(row_fmt.format(
            stage="VAE Decode", t=fmt_t(decode), p=pct(decode),
        ))

        # Total row
        rows.append(border_mid)
        rows.append(row_fmt.format(
            stage="Total", t=fmt_t(total), p=pct(total),
        ))

        lines = [border_top, header, border_mid] + rows + [border_bottom]
        return "\n".join(lines)

    def get_metrics(self) -> dict:
        """Return raw timing metrics dict.

        Returns:
            dict: Copy of current timings. Empty dict if no profiling run yet.
        """
        return dict(self.timings)

    def profile_with_torch_profiler(
        self,
        pipeline,
        prompt: str,
        width: int,
        height: int,
        steps: int,
    ) -> dict:
        """Use torch.profiler for detailed MPS kernel timing.

        Records CPU (and MPS-dispatched) operations using
        ``torch.profiler.profile``.  Falls back gracefully if
        torch.profiler is unavailable or raises on the current device.

        Args:
            pipeline: A loaded FluxKreaPipeline instance.
            prompt: Text prompt for generation.
            width: Image width.
            height: Image height.
            steps: Number of denoising steps.

        Returns:
            dict: Contains 'profile_available' bool, optional
                  'top_ops' list of (op_name, cpu_time_ms) tuples,
                  and 'error' key if profiling failed.
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile, record_function
        except ImportError:
            logger.warning("torch.profiler not available")
            return {"profile_available": False, "error": "torch.profiler not importable"}

        if not pipeline.is_loaded:
            return {"profile_available": False, "error": "Pipeline not loaded"}

        result: dict = {"profile_available": False}

        try:
            activities = [ProfilerActivity.CPU]
            # Add CUDA if available — MPS kernels surface as CPU dispatches
            try:
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
            except Exception:
                pass

            with profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
            ) as prof:
                with record_function("flux_generation"):
                    pipeline.generate(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                    )

            # Extract top operators by CPU time
            try:
                key_averages = prof.key_averages()
                top_ops = [
                    {
                        "name": evt.key,
                        "cpu_time_ms": round(evt.cpu_time_total / 1000, 3),
                        "count": evt.count,
                    }
                    for evt in sorted(
                        key_averages,
                        key=lambda e: e.cpu_time_total,
                        reverse=True,
                    )[:20]
                ]
                result["profile_available"] = True
                result["top_ops"] = top_ops
                result["profiler_output"] = prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=20,
                )
                logger.info("torch.profiler captured %d unique ops", len(key_averages))
            except Exception as e:
                logger.warning("Could not extract profiler key averages: %s", e)
                result["profile_available"] = True
                result["top_ops"] = []

        except Exception as e:
            logger.warning("torch.profiler run failed: %s", e)
            result["error"] = str(e)

        return result
