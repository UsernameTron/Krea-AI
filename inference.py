#!/usr/bin/env python3
import click
import torch
from pathlib import Path
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model
from src.flux.pipeline import Sampler


@click.command()
@click.option('--prompt', '-p', required=True, help='Text prompt for image generation')
@click.option('--width', '-w', default=1024, type=int, help='Image width (default: 1024)')
@click.option('--height', '-h', default=1024, type=int, help='Image height (default: 1024)')
@click.option('--guidance', '-g', default=4.5, type=float, help='Guidance scale (default: 4.5)')
@click.option('--num-steps', '-s', default=28, type=int, help='Number of sampling steps (default: 28)')
@click.option('--seed', default=42, type=int, help='Random seed (default: 42)')
@click.option('--output', '-o', default='output.png', help='Output image path (default: output.png)')
@click.option('--device', default='cuda', help='Device to use (default: cuda)')
def generate(prompt, width, height, guidance, num_steps, seed, output, device):
    torch_dtype = torch.bfloat16    
    click.echo("Loading models...")
    
    # Load models
    click.echo("Loading AE...")
    ae = load_ae("flux-krea-dev", device=device)

    click.echo("Loading CLIP...")
    clip = load_clip(device=device)
    
    click.echo("Loading T5...")
    t5 = load_t5(device=device)
    
    click.echo("Loading MMDiT...")
    model = load_flow_model("flux-krea-dev", device=device)
    model = model.to(device=device, dtype=torch_dtype)
    
    # Move models to device with specified dtype
    ae = ae.to(device=device, dtype=torch_dtype)
    clip = clip.to(device=device, dtype=torch_dtype)
    t5 = t5.to(device=device, dtype=torch_dtype)

    # Create sampler
    sampler = Sampler(
        model=model,
        ae=ae,
        clip=clip,
        t5=t5,
        device=device,
        dtype=torch_dtype,
    )
    
    click.echo(f"Generating image with prompt: '{prompt}'")
    click.echo(f"Parameters: {width}x{height}, guidance={guidance}, steps={num_steps}, seed={seed}")
    
    # Generate image
    image = sampler(
        prompt=prompt,
        width=width,
        height=height,
        guidance=guidance,
        num_steps=num_steps,
        seed=seed,
    )
    
    # Save image
    outpath = Path(output)
    image.save(outpath)
    
    click.echo(f"Image saved to: {outpath.absolute()}")


if __name__ == '__main__':
    generate()
