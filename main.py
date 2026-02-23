"""
Unified CLI entry point for FLUX.1 Krea.

Usage:
    python main.py generate --prompt "..." [--width 1024] [--height 1024] [--steps 28]
    python main.py web [--port 7860]
    python main.py benchmark [--quick]
    python main.py info
"""

import argparse
import logging
import sys

from config import FluxConfig, OptimizationLevel, get_config, get_hf_token


def cmd_generate(config: FluxConfig, args):
    """Generate an image from a text prompt."""
    from pipeline import FluxKreaPipeline

    pipeline = FluxKreaPipeline(config)

    print(f"Loading pipeline ({config.optimization_level} optimization)...")
    pipeline.load(progress_callback=lambda p, m: print(f"  [{p:.0%}] {m}"))

    if not pipeline.is_loaded:
        print("ERROR: Pipeline failed to load.", file=sys.stderr)
        sys.exit(1)

    seed = args.seed if args.seed >= 0 else None
    print(f"Generating {args.width}x{args.height}, {args.steps} steps, seed={seed or 'random'}...")

    image, metrics = pipeline.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=seed,
    )

    output_path = pipeline.save_image(image, args.prompt)
    print(
        f"Saved: {output_path}\n"
        f"Time: {metrics['generation_time']:.1f}s ({metrics['time_per_step']:.2f}s/step)\n"
        f"Device: {metrics['device']}, Seed: {metrics.get('seed', 'random')}"
    )

    pipeline.unload()


def cmd_web(config: FluxConfig, args):
    """Launch the Gradio web interface."""
    from app import create_ui

    if args.port:
        config.web.port = args.port
    if args.host:
        config.web.host = args.host
    if args.share:
        config.web.share = True

    print(f"Starting FLUX.1 Krea Studio on {config.web.host}:{config.web.port}")

    ui = create_ui(config)
    ui.launch(
        server_name=config.web.host,
        server_port=config.web.port,
        share=config.web.share,
    )


def cmd_benchmark(config: FluxConfig, args):
    """Run generation benchmarks."""
    from utils.benchmark import run_benchmark

    print(f"Running {'quick ' if args.quick else ''}benchmark...")
    results = run_benchmark(config=config, quick=args.quick)

    print("\nBenchmark Results:")
    print("-" * 50)
    for steps, data in sorted(results.items()):
        if "error" in data:
            print(f"  {steps} steps: ERROR - {data['error']}")
        else:
            print(
                f"  {steps} steps: {data['total_time']:.1f}s total, "
                f"{data['time_per_step']:.2f}s/step, "
                f"{data['pixels_per_second']:.0f} px/s"
            )


def cmd_info(config: FluxConfig, _args):
    """Show system information."""
    import torch

    from utils.monitor import get_system_info

    info = get_system_info()

    print("FLUX.1 Krea System Info")
    print("=" * 40)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {info.get('pytorch_version', '?')}")
    print(f"MPS:          {'Available' if info.get('mps_available') else 'Not available'}")
    print(f"CUDA:         {'Available' if info.get('cuda_available') else 'Not available'}")

    if info.get("system_memory_gb"):
        print(f"RAM:          {info['system_memory_gb']} GB ({info.get('memory_used_percent', '?')}% used)")
    if info.get("cpu_count"):
        print(f"CPU Cores:    {info.get('cpu_count_physical', '?')} physical, {info['cpu_count']} logical")
    if info.get("mps_allocated_mb"):
        print(f"MPS Memory:   {info['mps_allocated_mb']} MB allocated")

    print()
    print(f"HF Token:     {'Set' if get_hf_token() else 'NOT SET'}")
    print(f"Optimization: {config.optimization_level}")
    print(f"Device:       {config.device.preferred}")
    print(f"Model:        {config.model.id}")

    # Check if model is cached
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = config.model.id.replace("/", "--")
    model_cached = any(cache_dir.glob(f"models--{model_slug}*")) if cache_dir.exists() else False
    print(f"Model Cached: {'Yes' if model_cached else 'No'}")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1 Krea — Text-to-image generation for Apple Silicon",
    )
    parser.add_argument(
        "--optimization", "-O",
        choices=["none", "standard", "maximum"],
        help="Optimization level",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate an image")
    gen_parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    gen_parser.add_argument("--width", "-W", type=int, default=None, help="Image width (default: config)")
    gen_parser.add_argument("--height", "-H", type=int, default=None, help="Image height (default: config)")
    gen_parser.add_argument("--steps", "-s", type=int, default=None, help="Inference steps (default: config)")
    gen_parser.add_argument("--guidance-scale", "-g", type=float, default=None, help="Guidance scale")
    gen_parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 for random)")
    gen_parser.add_argument("--safety", action="store_true", help="Enable safety mode")

    # web
    web_parser = subparsers.add_parser("web", help="Launch web UI")
    web_parser.add_argument("--port", type=int, help="Server port")
    web_parser.add_argument("--host", type=str, help="Server host")
    web_parser.add_argument("--share", action="store_true", help="Create public link")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--quick", "-q", action="store_true", help="Quick benchmark (smaller images, fewer steps)")

    # info
    subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config with CLI overrides
    overrides = {}
    if args.optimization:
        overrides["optimization_level"] = args.optimization

    config = get_config(**overrides)

    # Apply generate-specific overrides
    if args.command == "generate":
        if args.width:
            config.generation.width = args.width
        if args.height:
            config.generation.height = args.height
        if args.steps:
            config.generation.steps = args.steps
        if args.guidance_scale:
            config.generation.guidance_scale = args.guidance_scale
        if args.safety:
            config.safety_mode.enabled = True

    commands = {
        "generate": cmd_generate,
        "web": cmd_web,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
    }

    try:
        commands[args.command](config, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal error: %s", e, exc_info=args.verbose)
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
