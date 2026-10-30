# RLinf OpenPI backend

`PI05Config(backend="rlinf")` selects RLinf's native PyTorch Gemma/SigLIP
network under the existing LeRobot `pi05` policy. The default remains
`backend="lerobot"`. Both backends use LeRobot's image preparation, tokenizer,
normalization, relative actions, action queue and remote RTC protocol.

## Provenance

`gemma.py`, `siglip.py`, `lora.py` and `utils.py` are vendored from
[RLinf/RLinf](https://github.com/RLinf/RLinf/tree/061c4cce80bbab998ef90585475d3f51290a2dfa/rlinf/models/embodiment/openpi/modules),
commit `061c4cce80bbab998ef90585475d3f51290a2dfa`, under Apache-2.0 (included in
`LICENSE`). Copyright notices are retained. The source tree at that revision
does not contain a NOTICE file.

`rlt.py` is vendored from the same revision's
[`rlinf/models/embodiment/modules/rlt_token_transformer.py`](https://github.com/RLinf/RLinf/blob/061c4cce80bbab998ef90585475d3f51290a2dfa/rlinf/models/embodiment/modules/rlt_token_transformer.py).
It retains the encoder, autoregressive decoder and parameter names, with local
per-sample loss reduction, an encoder-only export, and a forward encode mode
that respects PEFT/FSDP module wrappers.

Integration changes to those modules:

- Automatic compilation of individual kernels is disabled by default;
  `PI05Config.compile_model` controls compilation of the enclosing model.
- Adaptive RMSNorm also accepts `(batch, action_horizon, width)` conditioning
  for LeRobot's trained RTC. Scalar conditioning retains the upstream path.
- Formatting follows LeRobot's Ruff configuration.

`modeling.py` adapts network assembly and sampling to LeRobot. It reuses the
existing guided/trained RTC integrator. Time features are computed in FP32 and
cast to the configured projection dtype. The main repository's precision
profiles are retained: `lerobot` keeps the visual path in FP32, `openpi` uses
RLinf's selective BF16/FP32 precision, and `checkpoint` casts all VLA parameters to
the selected dtype. As in the original backend, the `openpi` profile or enabled
model compilation sets the process-wide FP32 matmul precision to `high`.
Euler integration and returned actions remain FP32.
The standalone RLinf observation
transforms and runtime are intentionally not imported. No extra RLinf, Ray,
Hydra, OpenPI or JAX installation is required.

## Checkpoints and supported features

The backend accepts native RLinf `img.*`/`llm.*` safetensors and converts
LeRobot/OpenPI PyTorch safetensors in memory, preserving the source dtype.
Conversion handles fused SigLIP QKV, transposed/stacked Gemma MLP weights,
RMSNorm/AdaRMS and tied input embeddings. Unused token-output heads and the
SigLIP pooling head are omitted. All other tensors must be recognized;
normal strict loading rejects missing parameters and incompatible shapes.

The source checkpoint is not modified. Saving the loaded policy produces a
LeRobot checkpoint with `backend="rlinf"` and native RLinf network keys under
`model.*`. For a bare RLinf checkpoint, supply an explicit `PI05Config` with
the matching action horizon/features; RLinf's bare `config.json` is not a
LeRobot policy config. Use the same LeRobot processors/stats as the baseline.

MEM configuration is rejected rather than silently discarded. Standard
forward training, expert/vision freezing, gradient checkpointing, guided RTC,
trained RTC and PEFT are implemented; full-size model validation is separate.
Sampling uses LeRobot's existing compilation dispatcher. Fast guided RTC
prepares prefix padding, masks and delay controls outside the compiled sampler,
so changing delays/prefix lengths can reuse its graph at fixed observation
shapes and inference-step count. Native/reference RTC retains separate compiled
prefix and denoiser regions.

RLT Stage 1 is opt-in with `use_rlt=True`: `frozen_vla` trains only the RLT
encoder/decoder on detached final-layer VLM prefix features; `joint` adds
`rlt_alpha * vla_loss` and supports the existing mixed LoRA/full-expert recipe.
New RLT weights remain FP32; normal outer AMP controls compute precision.
The existing `gradient_checkpointing` flag also covers the RLT encoder and
decoder during training, including PEFT copies and frozen-VLA training.
Only a declared ordinary LeRobot PI05 base may initialize a new RLT
automatically. Bare native bases require `initialize_rlt=True` in the loading
API. Partial RLT weights, or an RLT-enabled saved config missing its entire
RLT module, are rejected even when ordinary core loading is non-strict.
RLT structure must match the saved configuration. Other RLinf task-specific
heads remain unsupported and strict loading rejects them.

`PI05Policy.extract_rlt_features(preprocessed_batch)` returns `(batch, rlt_embed_dim)`.
It uses the saved normalization/tokenization and the same final prefix as
training. `examples/tutorial/pi05/export_rlt.py` exports the VLA, RLT encoder,
and processors; it merges PEFT checkpoints and omits the reconstruction decoder.
Encoder-only exports sample actions and extract features but reject training.
Keep the complete training checkpoint to resume Stage 1. See `docs/source/pi05.mdx`
for commands. Stage 2 actor-critic, online reward/replay and RTC transition
collection are not part of Stage 1.

`PI05Config(peft_mode="vlm_lora_expert_full")` selects LoRA on language and
vision attention and MLPs, while fully training/saving the action expert, its
adaptive norms, action projections and time MLPs. All VLM base tensors remain
frozen. `peft_language_lora` and `peft_vision_lora` independently select `none`,
`attention` or `attention_mlp` (the default) for each component. Setting both to
`none` trains/saves only the full expert. Language MLP adapters apply separate
gate/up/down low-rank projections to activations and merge into native
`w_gating`/`w_linear` checkpoint tensors.
The fused vision adapter in `peft.py` uses temporary effective weights rather
than merging/unmerging each forward, avoiding frozen-weight drift in BF16.
Permanent merging applies the output delta once. The policy's PEFT preparation
hook installs this custom dispatch during initial wrapping and `make_policy`
reloads. The raw converter accepts full/merged weights; it does not convert
unmerged adapters between network layouts.
The fused vision adapter requires `lora_dropout=0` and `lora_bias=false`;
nonzero LoRA dropout and LoRA-B biases are not implemented for its effective
weight path. Initial wrapping, checkpoint loading and adding adapters reject
these settings explicitly. This restriction is local to fused vision attention;
the native language MLP adapter retains its activation dropout support.

Use `examples/tutorial/pi05/merge_lora.py` to export an adapter and its full
trained expert as an ordinary LeRobot checkpoint with both saved processors
for Piper/RTC inference. Retain the base checkpoint while using an adapter.

RLinf retains exact GELU, while the reference OpenPI uses tanh-approximate
GELU. This is a known intentional numerical difference (see
[RLinf issue #1725](https://github.com/RLinf/RLinf/issues/1725)). Backend
comparisons should measure differences rather than assume bitwise parity.
Conversion parity tests align the source activation to isolate weight-layout
conversion; they do not establish equivalence to an unchanged LeRobot model.
