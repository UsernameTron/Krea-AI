"""
Unified Gradio web interface for FLUX.1 Krea.

Replaces: flux_web_ui_official.py, flux_web_simple.py,
          flux_web_m4_optimized.py, flux_web_timeout_protected.py, flux_web_debug.py
"""

import logging
import signal
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from config import FluxConfig, OptimizationLevel, get_config, get_hf_token
from pipeline import FluxKreaPipeline

logger = logging.getLogger(__name__)

# Debug log buffer
_debug_log: deque = deque(maxlen=50)


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    _debug_log.append(entry)
    logger.info(msg)


class FluxWebApp:
    """Unified Gradio web application for FLUX.1 Krea."""

    def __init__(self, config: FluxConfig):
        self.config = config
        self.pipeline = FluxKreaPipeline(config)
        self._lock = threading.Lock()
        self._load_event = threading.Event()
        self._load_error: str = ""
        self._session_gallery: list = []

    def _background_load(self):
        """Load pipeline in background thread."""
        try:
            _log("Loading pipeline...")
            self.pipeline.load(progress_callback=lambda p, m: _log(f"[{p:.0%}] {m}"))
            _log("Pipeline loaded successfully")
        except Exception as e:
            self._load_error = str(e)
            _log(f"Pipeline load failed: {e}")
        finally:
            self._load_event.set()

    def start_loading(self):
        """Start background pipeline loading."""
        thread = threading.Thread(target=self._background_load, daemon=True)
        thread.start()

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        safety_mode: bool,
        timeout_minutes: int,
        progress=gr.Progress(),
    ):
        """Generate an image with concurrency protection and timeout."""
        if not prompt.strip():
            return None, "Please enter a prompt.", self._get_debug_text()

        # Wait for pipeline to finish loading
        if not self._load_event.is_set():
            progress(0.05, desc="Waiting for pipeline to load...")
            self._load_event.wait(timeout=300)

        if self._load_error:
            return None, self._format_error(self._load_error), self._get_debug_text()

        if not self.pipeline.is_loaded:
            return None, "Pipeline failed to load. Check debug log.", self._get_debug_text()

        # Concurrency lock
        if not self._lock.acquire(blocking=False):
            return None, "Another generation is in progress. Please wait.", self._get_debug_text()

        try:
            # Apply safety mode override
            if safety_mode:
                self.config.safety_mode.enabled = True
            else:
                self.config.safety_mode.enabled = False

            seed_val = seed if seed >= 0 else None
            _log(f"Generating: {width}x{height}, {steps} steps, guidance={guidance_scale}, seed={seed_val}")

            progress(0.1, desc="Generating image...")

            # Timeout protection via signal (Unix only)
            timed_out = False
            old_handler = None
            try:
                def _timeout_handler(signum, frame):
                    nonlocal timed_out
                    timed_out = True
                    raise TimeoutError("Generation timed out")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout_minutes * 60)
            except (ValueError, AttributeError):
                pass  # Not on Unix or not main thread

            try:
                image, metrics = self.pipeline.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    seed=seed_val,
                )
            finally:
                try:
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                except (ValueError, AttributeError):
                    pass

            if timed_out:
                return None, f"Generation timed out after {timeout_minutes} minutes.", self._get_debug_text()

            # Save image
            output_path = self.pipeline.save_image(image, prompt)
            self._session_gallery.append(str(output_path))

            progress(1.0, desc="Done!")

            # Format metrics
            info = (
                f"Generated in {metrics['generation_time']:.1f}s "
                f"({metrics['time_per_step']:.2f}s/step) | "
                f"{width}x{height} | {steps} steps | "
                f"guidance={guidance_scale} | "
                f"seed={metrics.get('seed', 'random')} | "
                f"device={metrics['device']}"
            )
            _log(info)

            return image, info, self._get_debug_text()

        except TimeoutError:
            _log(f"Generation timed out after {timeout_minutes} minutes")
            return None, f"Generation timed out after {timeout_minutes} minutes.", self._get_debug_text()

        except Exception as e:
            _log(f"Generation error: {e}")
            return None, self._format_error(str(e)), self._get_debug_text()

        finally:
            self._lock.release()

    def _format_error(self, error_msg: str) -> str:
        """Format error with actionable solutions."""
        msg = f"Error: {error_msg}\n\n"

        if "gated" in error_msg.lower():
            msg += (
                "SOLUTION — Repository Access Required:\n"
                "1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev\n"
                "2. Click 'Request access to this repository'\n"
                "3. Accept license agreement\n"
                "4. Set HF_TOKEN environment variable"
            )
        elif "401" in error_msg or "403" in error_msg:
            msg += (
                "SOLUTION — Authentication:\n"
                "1. Go to: https://huggingface.co/settings/tokens\n"
                "2. Create token with 'Read' permissions\n"
                "3. Export: export HF_TOKEN='your_token_here'"
            )
        elif "memory" in error_msg.lower() or "oom" in error_msg.lower():
            msg += (
                "SOLUTION — Memory:\n"
                "1. Reduce image resolution (try 768x768)\n"
                "2. Reduce inference steps\n"
                "3. Close other applications\n"
                "4. Enable safety mode for conservative memory usage"
            )
        elif "timeout" in error_msg.lower():
            msg += (
                "SOLUTION — Timeout:\n"
                "1. Increase timeout in settings\n"
                "2. Reduce image size or steps\n"
                "3. Use 'none' optimization level"
            )

        return msg

    def _get_debug_text(self) -> str:
        return "\n".join(_debug_log) if _debug_log else "No debug messages yet."

    def _get_system_info_text(self) -> str:
        info = self.pipeline.get_system_info()
        lines = [
            f"PyTorch: {info.get('pytorch_version', '?')}",
            f"Device: {info.get('device', '?')}",
            f"MPS: {'Available' if info.get('mps_available') else 'Not available'}",
            f"Memory: {info.get('system_memory_gb', '?')} GB",
            f"HF Token: {'Set' if info.get('hf_token_set') else 'NOT SET'}",
            f"Optimization: {info.get('optimization_level', '?')}",
            f"Loaded: {'Yes' if info.get('is_loaded') else 'No'}",
        ]
        if info.get("mps_allocated_mb"):
            lines.append(f"MPS Memory: {info['mps_allocated_mb']} MB")
        return "\n".join(lines)


def create_ui(config: FluxConfig) -> gr.Blocks:
    """Create the unified Gradio interface."""
    app = FluxWebApp(config)

    with gr.Blocks(
        title="FLUX.1 Krea Studio",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 10px; }
        .debug-log { font-family: monospace; font-size: 12px; }
        """,
    ) as ui:
        gr.Markdown(
            "# FLUX.1 Krea Studio\n"
            "Text-to-image generation optimized for Apple Silicon"
        )

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                )

                with gr.Accordion("Settings", open=False):
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=config.generation.width, step=64, label="Width")
                        height = gr.Slider(512, 1536, value=config.generation.height, step=64, label="Height")
                    steps = gr.Slider(4, 50, value=config.generation.steps, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 10.0, value=config.generation.guidance_scale, step=0.5, label="Guidance Scale")
                    seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    safety_mode = gr.Checkbox(value=config.safety_mode.enabled, label="Safety Mode (prevents black images)")
                    timeout = gr.Slider(1, 15, value=5, step=1, label="Timeout (minutes)")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                info_box = gr.Textbox(label="Generation Info", interactive=False)

            # Right column: output
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")

        # Debug section
        with gr.Accordion("Debug Log & System Info", open=False):
            with gr.Row():
                with gr.Column():
                    system_info = gr.Textbox(
                        label="System Info",
                        value=app._get_system_info_text(),
                        interactive=False,
                        lines=8,
                    )
                    refresh_btn = gr.Button("Refresh System Info", size="sm")
                with gr.Column():
                    debug_log = gr.Textbox(
                        label="Debug Log",
                        value=app._get_debug_text(),
                        interactive=False,
                        lines=8,
                        elem_classes=["debug-log"],
                    )

        # Tips
        gr.Markdown(
            "**Tips:** Use 768x768 for faster generation. "
            "Enable Safety Mode if you get black images. "
            "Seed -1 means random."
        )

        # Wire up events
        generate_btn.click(
            fn=app.generate,
            inputs=[prompt, width, height, steps, guidance, seed, safety_mode, timeout],
            outputs=[output_image, info_box, debug_log],
        )
        refresh_btn.click(
            fn=lambda: (app._get_system_info_text(), app._get_debug_text()),
            outputs=[system_info, debug_log],
        )

        # Start loading pipeline on UI launch
        ui.load(fn=app.start_loading)

    return ui


def main():
    """Launch the web UI."""
    import argparse

    parser = argparse.ArgumentParser(description="FLUX.1 Krea Web UI")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--host", type=str, help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    config = get_config()
    if args.port:
        config.web.port = args.port
    if args.host:
        config.web.host = args.host
    if args.share:
        config.web.share = True

    _log(f"Starting FLUX.1 Krea Studio on {config.web.host}:{config.web.port}")

    ui = create_ui(config)
    ui.launch(
        server_name=config.web.host,
        server_port=config.web.port,
        share=config.web.share,
    )


if __name__ == "__main__":
    main()
