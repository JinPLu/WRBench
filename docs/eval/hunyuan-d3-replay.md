# Reproduce Qwen D3 and D4 on released videos

These replays trace three D3 probes for Hunyuan WorldPlay and three D4 probes for LiveWorld. Download the complete MP4s and preserve each row's `world_state_prompt`. [Issue #11](https://github.com/JinPLu/WRBench/issues/11) · [Pinned score index](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/index/videos_master.jsonl).

| Example | `video_id` | MP4 SHA-256 |
| --- | --- | --- |
| [Hunyuan WorldPlay D3](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/videos/hunyuan_worldplay/static/hunyuan_worldplay__living_room_adult_sofa_sit__T2_div_b__none__static.mp4) | `hunyuan_worldplay__living_room_adult_sofa_sit__T2_div_b__none__static` | `c651a8b18a9950e2c775860b28ee1cb3170ff18de0ef7e029b1e88bc9b864769` |
| [LiveWorld D4](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/videos/liveworld/static/liveworld__library_student_chair_sit__T0__none__static.mp4) | `liveworld__library_student_chair_sit__T0__none__static` | `ba86644ecc4446c994a4fee0bcee2de81150bcb27cafd624976bc08fc40d96db` |

## Replay settings

For comparison with released scores, start with the historical replay column and use the pinned model snapshot for weights, tokenizer, chat template and processor. Set `eval.scorers.qwen_scorer_python` to that environment's interpreter and `eval.scorers.qwen35_model` to the pinned model directory. `wrbench_default` selects benchmark scoring settings such as the prompt profile and frame rate. The scorer environment determines how the processor expands the video into input tokens. Save the environment versions with your scores. The token-alignment replay below isolates the effect of the extra tokens under the issue environment.

Both examples ran on the same NVIDIA H20-3e in two environments. They shared Python 3.11.15, bfloat16, FlashAttention 2.8.3, einops 0.8.2, decord 0.6.0 and Accelerate 1.16.0. The scorer used the P25 probe profile, `task_context_mode=none`, 2 fps, one scoring sample and the processor's default pixel budgets. The Gated DeltaNet layers used the PyTorch fallback in both environments. [WRBench scorer at `51e5c91`](https://github.com/JinPLu/WRBench/tree/51e5c914fd935fc0661b921bafebccb66e31b104) · [Qwen3.5-9B weights and processor files at `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a).

| Component | Historical replay | Issue replay |
| --- | --- | --- |
| Transformers | 5.9.0 | 5.22.1 |
| PyTorch | 2.11.0+cu128 | 2.8.0+cu128 |
| torchvision | 0.26.0+cu128 | 0.23.0+cu128 |
| tokenizers | 0.22.2 | 0.23.3 |
| FlashAttention 2.8.3 install | Source build for `sm_90` with CUDA toolkit 12.9 | [Upstream CPython 3.11 wheel for Torch 2.8](https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl) |
| Model placement | `from_pretrained` with `device_map={"": 0}` | Load on CPU, then move to `cuda:0` |

The archived run records the historical Transformers and PyTorch versions, BF16 and the FlashAttention2 backend. Its exact FlashAttention and decord builds were unavailable, so the table specifies the versions used for the replay. Get PyTorch and torchvision from the [CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128). For the Torch 2.11 source build, set `CUDA_HOME` to the installed CUDA toolkit and set `FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90` when [installing FlashAttention 2.8.3 from source](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/README.md#installation-and-features).

To run the public P25 scorer, save a JSON array containing the single index row as `sample.json`, with `video_path` pointing to the downloaded file. Set `MODEL_DIR` to the pinned checkpoint directory and choose a fresh `OUT_DIR` for each environment. The command writes probe probabilities to `raw_v2_probe_scores.jsonl`.

```bash
WORLD_STATE_VIDEO_BACKEND=decord \
python -m wrbench.eval.scoring.run_local_qwen35_probe_logprob_scorer \
  --manifest-path sample.json --model-path "$MODEL_DIR" --output-dir "$OUT_DIR" \
  --experiment-id qwen-probe-replay --vlm-name qwen35 --loader-family auto \
  --prompt-mode runtime_v2_probe_logprob_p25_d3d4_slot_parse --task-context-mode none \
  --fps 2 --dtype bfloat16 --attn-implementation flash_attention_2 \
  --num-shards 1 --shard-id 0 --local-rank 0 --max-videos 1 --num-samples 1 \
  --progress-every 1 --evidence-context-mode visibility_v1
```

The public command uses explicit CUDA placement and runs all 16 active probes. The traces below select the three probes for the target metric from the same scorer and use the placement recorded in the table.

WRBench uses the supported Qwen3.5 multimodal input path. The processor's `apply_chat_template` prepares the video and text together, with `enable_thinking=False` for direct answers. WRBench defines the scoring protocol. One forward pass supplies the next-token logits for `p_yes`, the probability of the Yes variants conditional on the Yes and No variants. Both metrics average the two positive probe probabilities and `1 - p_yes` for the negative probe. Qwen's temperature, top-p and top-k recommendations apply to generated replies. [Transformers model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) · [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#instruct-or-non-thinking-mode).

## Hunyuan WorldPlay D3

Both processors sampled the same zero-based frame indices `[0, 14, 28, 41, 55, 69, 83, 96, 110, 124]` from the 125-frame, 24 fps MP4. Their decoded frame bytes matched. Both produced a resized height and width of 448 × 832, `video_grid_thw=[5, 28, 52]` and the same `float32` video tensor of shape `[7280, 1536]` (SHA-256 `00609f65afb058823fafb995a6e2971f32955a59d626b8f8d2a5dde653548430`).

The probe text and rendered chat matched before video-token expansion. For each probe, Transformers 5.22.1 added exactly two tokens to the 5.9.0 sequence, `<|vision_start|>` (ID `248053`) at zero-based index 3 and `<|vision_end|>` (ID `248054`) at index 1864. In probe order `D3_POSITION_RELATION`, `D3_STATIC_ANCHORS`, `D3_NEG_WRONG_SPATIAL`, the lengths changed from 2,333, 2,318 and 2,326 to 2,335, 2,320 and 2,328. The older processor replaced the full wrapped video placeholder. The newer processor expands the inner placeholder and retains its surrounding markers. [Transformers 5.9.0 processor](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py), [Transformers 5.22.1 processor](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py).

The two tokens also shift the final multimodal rotary position by two. Removing them from the input IDs, attention mask and modality IDs restored the Transformers 5.9.0 replay's tensor hashes for all three probes, including the position IDs computed from those inputs.

| Probe or metric | Historical archive | Historical replay | Issue replay | Issue replay with aligned tokens |
| --- | ---: | ---: | ---: | ---: |
| `D3_POSITION_RELATION` `p_yes` | 0.095417 | 0.106761 | 0.053473 | 0.095420 |
| `D3_STATIC_ANCHORS` `p_yes` | 0.348826 | 0.348812 | 0.182576 | 0.377707 |
| `D3_NEG_WRONG_SPATIAL` `p_yes` | 0.245249 | 0.222852 | 0.438112 | 0.222846 |
| **D3** | **0.399665** | **0.410907** | **0.265979** | **0.416760** |

The historical replay is 0.011242 above the released 0.399665. The issue replay is 0.003844 above the reported 0.262135. Within a single model load in the issue environment, removing the two markers increased D3 by 0.150781 and brought it within 0.005853 of the historical replay. This identifies the extra markers as the main cause of the D3 gap between the two replays on this video. `D3_STATIC_ANCHORS` still differs by 0.028895 after alignment. A broader reproduction tolerance requires more videos and runtime comparisons. [Published score](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/index/videos_master.jsonl), [issue report](https://github.com/JinPLu/WRBench/issues/11).

## LiveWorld D4

The index's task is `A reader stands in the reading room, facing a reading chair.` Both environments sampled frames `[0, 9, 18, 27, 37, 46, 55, 64]` from the 65-frame, 16 fps MP4. Their decoded frames and resized 480 × 832 inputs matched, with `video_grid_thw=[4, 30, 52]` and identical `float32` video tensors of shape `[6240, 1536]` (SHA-256 `928c2ca5db2e600650dceefad1c785332adaca7133b3374a173267148b9412b7`).

The rendered chat again matched before video-token expansion. Transformers 5.22.1 retained the same extra boundary token IDs, `248053` and `248054`, at zero-based indices 3 and 1596. In table order, input lengths changed from 2,066, 2,055 and 2,057 to 2,068, 2,057 and 2,059. Removing those two positions restored the 5.9.0 replay's input tensor and rotary position ID hashes for all three probes.

| Probe or metric | Historical archive | Historical replay | Issue replay | Issue replay with aligned tokens |
| --- | ---: | ---: | ---: | ---: |
| `D4_ACTION_STATE` `p_yes` | 0.731183 | 0.731186 | 0.531518 | 0.705934 |
| `D4_TEMPORAL_PROGRESS` `p_yes` | 0.817646 | 0.817647 | 0.622688 | 0.798265 |
| `D4_NEG_ABSENT_FREEZE` `p_yes` | 0.060221 | 0.060223 | 0.053561 | 0.067702 |
| **D4** | **0.829536** | **0.829537** | **0.700215** | **0.812166** |

The historical replay is 0.000001 above the released 0.829536. The issue replay is 0.023731 below the reported 0.723946. Removing the two boundary tokens within a single model load recovered 0.111951 of the 0.129322 gap between the replays (86.6%). The remaining gap is 0.017371. The two examples identify the same primary mechanism for these D3 and D4 replays. D5/D6 and a general reproduction tolerance require separate measurements. [Published score](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/index/videos_master.jsonl) · [Issue report](https://github.com/JinPLu/WRBench/issues/11).
