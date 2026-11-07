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

## Documentation

This directory contains the vendored network and its LeRobot integration. Detailed
implementation and compatibility notes are maintained with the project documentation:

- [RLinf backend and RLT architecture](../../../../../docs/architecture/rlt.md).
- [RLT training, export, replay learning and remote collection commands](../../../../../docs/workflows/rlt/README.md).
- [Checkpoint, LoRA, feature identity and runtime limits](../../../../../docs/troubleshooting/rlt.md).
- [Complete Pi0.5 website tutorial](../../../../../docs/source/pi05.mdx#optional-rlinf-network-backend).

Keep this README and LICENSE with the complete directory when synchronizing or
packaging the backend. The ordinary `pi05` policy remains the integration entrypoint.
