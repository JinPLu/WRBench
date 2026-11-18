"""Run official ReCamMaster from a WRBench camera embedding."""

# Adapted from the ReCamMaster reference inference code (MIT).
# MIT License
#
# Copyright (c) 2025 Kuaishou Visual Generation and Interaction Center
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory


def _bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--wan-model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--camera-npy", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tiled", type=_bool, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision.transforms import InterpolationMode, functional
    from torchvision.transforms import v2
    from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video

    transform = v2.Compose([
        v2.CenterCrop((args.height, args.width)),
        v2.Resize((args.height, args.width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    frames = []
    with imageio.get_reader(args.source_video) as reader:
        for frame in islice(reader, args.num_frames):
            image = Image.fromarray(frame).convert("RGB")
            width, height = image.size
            scale = max(args.width / width, args.height / height)
            image = functional.resize(
                image, (round(height * scale), round(width * scale)),
                interpolation=InterpolationMode.BILINEAR,
            )
            frames.append(transform(image))
    if len(frames) != args.num_frames:
        raise ValueError(f"source video contains {len(frames)} decoded frames; need at least {args.num_frames}")
    source_video = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)
    target_camera = torch.from_numpy(np.load(args.camera_npy, allow_pickle=False)).to(torch.bfloat16)

    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([
        str(args.wan_model_dir / "diffusion_pytorch_model.safetensors"),
        str(args.wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(args.wan_model_dir / "Wan2.1_VAE.pth"),
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(manager, device="cuda")
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
    pipe.dit.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    pipe.to(device="cuda", dtype=torch.bfloat16)
    video = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        source_video=source_video, target_camera=target_camera,
        cfg_scale=args.cfg_scale, num_inference_steps=args.num_inference_steps,
        seed=args.seed, height=args.height, width=args.width,
        num_frames=args.num_frames, tiled=args.tiled,
    )
    if len(video) != args.num_frames:
        raise RuntimeError(f"ReCamMaster returned {len(video)} frames; expected {args.num_frames}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        encoded = Path(directory) / "video.mp4"
        save_video(video, str(encoded), fps=args.fps, quality=5)
        if not encoded.stat().st_size:
            raise RuntimeError("ReCamMaster produced an empty MP4")
        encoded.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
