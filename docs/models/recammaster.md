# ReCamMaster

ReCamMaster re-renders a source video along a requested camera path. WRBench
uses the first 81 decoded frames, resizes and center-crops them to 832 × 480,
and writes an 81-frame MP4 at 16 fps by default. WRBench compiles the camera
request into 21 poses relative to the first pose, in the `(1, 21, 12)` embedding
expected by the renderer.

## Compile a camera request

Run the commands from the WRBench checkout. Set `SOURCE` to any MP4 with at
least 81 frames and `PROMPT` to a description of its content. This example
assumes the upstream checkout is beside WRBench. [Upstream sample and captions](https://github.com/KlingAIResearch/ReCamMaster/tree/fcf98bc86e876bb534518cd99e8a65b282f0f16e/example_test_data)

```bash
SOURCE="../ReCamMaster/example_test_data/videos/1.mp4"
PROMPT="Two people dance on a city street at dusk."

wrbench generate --model recammaster --camera preset:yaw_LR --peak-deg 30 \
  --source-video "$SOURCE" --prompt "$PROMPT" --out out/recammaster.mp4
```

This writes the target trajectory and sidecars, including
`out/recammaster.mp4.payload.json`. The compile step runs without loading model
weights. Try another preset with `wrbench presets`, or supply a camera script
using the [camera-control grammar](../camera-control.md).

## Run on CUDA

1. Install the upstream ReCamMaster checkout and its dependencies in a separate
   CUDA-capable Python environment. [Upstream inference setup](https://github.com/KlingAIResearch/ReCamMaster/blob/fcf98bc86e876bb534518cd99e8a65b282f0f16e/README.md#inference)
2. Download `step20000.ckpt` and the full Wan2.1-T2V-1.3B snapshot. Keep the Wan
   weights and their `google/umt5-xxl` tokenizer in the same snapshot layout.
   [ReCamMaster checkpoint](https://huggingface.co/KlingTeam/ReCamMaster-Wan2.1/blob/main/step20000.ckpt) · [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
3. Copy [the runtime template](../../wrbench.runtime.example.json) to
   `wrbench.runtime.json`. In `models.recammaster`, set `python_bin` to the
   upstream environment's Python, `repo_root` to the upstream checkout,
   `model_path` to `step20000.ckpt`, and `extra_paths.wan_model_dir` to the Wan
   snapshot. Use absolute paths and set `gpu_id` for your machine. The template
   supplies inference settings, including `negative_prompt: ""` for an empty
   negative prompt.

Run the same request with the runtime file and real generation enabled.

```bash
wrbench generate --model recammaster --camera preset:yaw_LR --peak-deg 30 \
  --source-video "$SOURCE" --prompt "$PROMPT" --out out/recammaster.mp4 \
  --runtime-config wrbench.runtime.json --no-dry-run
```

The subprocess passes WRBench's camera embedding to the upstream pipeline and
writes the MP4 alongside the sidecars. Generation requires 81 frames at
832 × 480. Other dimensions and frame counts are available for compile-only
inspection. `--fps` sets output playback speed without resampling the source
frames.

If the output video already exists, choose a new `--out` path to keep the
existing video paired with its camera sidecars.

Use `wrbench doctor --model recammaster` to check the local adapter and compile
path. The generation command validates the configured runtime paths before
launching.
