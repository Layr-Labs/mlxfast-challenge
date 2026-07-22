# Laguna Transformed-Weight Contract

Interface between the offline transform (`Sources/MLXFastTransform/`) and the
Laguna runtime (`LagunaConfig` / `LagunaWeightLoader` /
`LagunaRuntimeWeightCache` / `LagunaRuntimeModel` in this directory). The
transform must produce a weights directory that satisfies everything below;
the runtime validates all of it (`LagunaWeightLoader.validateRequiredMetadata`
plus the shard/tensor inventory checks in `loadRuntimeWeightValues`) before
the first forward.

Source checkpoint: `mlx-community/Laguna-XS-2.1-4bit` (Poolside Laguna XS
2.1, 256-expert MoE, MLX affine quantization). All shapes, dtypes, and
per-tensor quantization below were verified against the pinned checkpoint's
safetensors headers and `config.json`.

## Directory layout

```text
weights/
  config.json                      runtime config (schema parsed by LagunaConfig.load)
  model.safetensors.index.json     standard HF index: {"weight_map": {tensor -> shard}}
  model-XXXXX-of-YYYYY.safetensors one or more shards
```

- Every shard referenced by the index must exist, every `*.safetensors` file
  present must be referenced, and each shard must contain exactly the tensors
  the index maps to it (validated both directions at load).
- Tensor names must be unique across shards, and must remain unique after the
  runtime strips the `language_model.` prefix.

## Naming scheme

The transform keeps the source checkpoint's tensor names **unchanged**,
including the `language_model.` text-tower prefix (`LagunaWeightNames`
addresses tensors by these names). At load time the runtime renames
`language_model.X` to `X` (`RuntimeWeightNameTracker`), which aligns the
tensors with the module parameter paths of `LagunaRuntimeModel`
(`model.layers.N...`, `lm_head.*`).

Quantized tensors are affine-packed triplets: `<stem>.weight` (U32 packed
codes), `<stem>.scales` (BF16), `<stem>.biases` (BF16). A stem is quantized
exactly when its `.scales` companion exists; the runtime derives bits from
the packed width, so the layout below is the source of truth.

## Quantization

Global: affine, `group_size = 64`, `bits = 4` — declared in `config.json`
under `quantization` (and mirrored in `quantization_config`) as
`{"group_size": 64, "bits": 4, "mode": "affine"}`.

Per-tensor overrides: the same `quantization` object additionally maps the 39
router-gate stems to 8-bit:

```json
"language_model.model.layers.<N>.mlp.gate.proj": {"group_size": 64, "bits": 8}
```

for every sparse layer `N` in 1...39. No other overrides exist.

For a quantized stem with logical shape `[rows, in]` (or `[experts, rows,
in]`), the stored shapes are:

```text
<stem>.weight  U32  [..., rows, in * bits / 32]
<stem>.scales  BF16 [..., rows, in / 64]
<stem>.biases  BF16 [..., rows, in / 64]
```

## Full tensor inventory (1634 tensors)

Geometry constants (see `LagunaConstants`): hidden 2048, vocab 100352, 40
layers, head_dim 128, 8 KV heads, per-layer query heads 48 (full-attention
layers 0, 4, 8, ..., 36) / 64 (the 30 sliding layers), dense intermediate
8192 (layer 0 only), 256 experts, moe intermediate 512, shared expert 512.

### Top level (7 tensors)

| Tensor | Quant | Logical shape | Stored weight/scales shapes |
| --- | --- | --- | --- |
| `language_model.model.embed_tokens.{weight,scales,biases}` | 4-bit g64 | [100352, 2048] | [100352, 256] / [100352, 32] |
| `language_model.lm_head.{weight,scales,biases}` (UNTIED) | 4-bit g64 | [100352, 2048] | [100352, 256] / [100352, 32] |
| `language_model.model.norm.weight` | BF16 | [2048] | — |

### Per layer, all 40 layers (19 tensors each = 760)

Prefix `language_model.model.layers.<N>.`; `H(N)` = 48 on full-attention
layers, 64 on sliding layers.

| Tensor | Quant | Logical shape |
| --- | --- | --- |
| `input_layernorm.weight` | BF16 | [2048] |
| `post_attention_layernorm.weight` | BF16 | [2048] |
| `self_attn.q_proj.{weight,scales,biases}` | 4-bit g64 | [H(N)*128, 2048] |
| `self_attn.k_proj.{weight,scales,biases}` | 4-bit g64 | [1024, 2048] |
| `self_attn.v_proj.{weight,scales,biases}` | 4-bit g64 | [1024, 2048] |
| `self_attn.o_proj.{weight,scales,biases}` | 4-bit g64 | [2048, H(N)*128] |
| `self_attn.g_proj.{weight,scales,biases}` (per-head gate) | 4-bit g64 | [H(N), 2048] |
| `self_attn.q_norm.weight` | BF16 | [128] |
| `self_attn.k_norm.weight` | BF16 | [128] |

Examples of stored shapes: layer 0 `q_proj.weight` U32 [6144, 256], layer 1
`q_proj.weight` U32 [8192, 256]; layer 0 `o_proj.weight` U32 [2048, 768],
layer 1 `o_proj.weight` U32 [2048, 1024]; `g_proj.weight` U32 [48, 256] /
[64, 256].

### Layer 0 only — dense MLP (9 tensors)

| Tensor | Quant | Logical shape |
| --- | --- | --- |
| `mlp.gate_proj.{weight,scales,biases}` | 4-bit g64 | [8192, 2048] |
| `mlp.up_proj.{weight,scales,biases}` | 4-bit g64 | [8192, 2048] |
| `mlp.down_proj.{weight,scales,biases}` | 4-bit g64 | [2048, 8192] |

### Layers 1–39 — sparse MoE (22 tensors each = 858)

| Tensor | Quant | Logical shape | Stored weight shape |
| --- | --- | --- | --- |
| `mlp.gate.proj.{weight,scales,biases}` (router) | **8-bit** g64 | [256, 2048] | U32 [256, 512], scales [256, 32] |
| `mlp.gate.e_score_correction_bias` | BF16 | [256] | — |
| `mlp.switch_mlp.gate_proj.{weight,scales,biases}` | 4-bit g64 | [256, 512, 2048] | U32 [256, 512, 256], scales [256, 512, 32] |
| `mlp.switch_mlp.up_proj.{weight,scales,biases}` | 4-bit g64 | [256, 512, 2048] | U32 [256, 512, 256] |
| `mlp.switch_mlp.down_proj.{weight,scales,biases}` | 4-bit g64 | [256, 2048, 512] | U32 [256, 2048, 64], scales [256, 2048, 8] |
| `mlp.shared_expert.gate_proj.{weight,scales,biases}` | 4-bit g64 | [512, 2048] | U32 [512, 256] |
| `mlp.shared_expert.up_proj.{weight,scales,biases}` | 4-bit g64 | [512, 2048] | U32 [512, 256] |
| `mlp.shared_expert.down_proj.{weight,scales,biases}` | 4-bit g64 | [2048, 512] | U32 [2048, 64] |

The `switch_mlp` tensors are the SwitchGLU stacked-expert layout (leading
axis = expert index) that `mlx_lm` writes and the vendored
`SwitchGLU`/`QuantizedSwitchLinear` consume directly. The transform must NOT
split them into per-expert tensors.

Count check: 7 + 40*19 + 9 + 39*22 = 1634.

## What must NOT be present

- No tensors outside the `language_model.` prefix (the source checkpoint's
  empty `vision_config` contributes none).
- No `rotary_emb.inv_freq` tables (the runtime would drop them via
  `sanitize`, but the inventory check runs first — leave them out).
- No fused/derived layouts in v1: no `gate_up_proj` fusion, no transposed
  experts, no extra metadata sidecars. Derived runtime layouts are a later
  optimization wave and will extend this contract explicitly.

## config.json requirements

`LagunaConfig.load` parses the flat source-config schema; the transform may
copy the source `config.json` fields directly (defaults equal the pinned
checkpoint's values, so trimmed configs also parse). Load-bearing fields:

- `model_type` ("laguna"), `vocab_size`, `hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`,
  `num_attention_heads_per_layer` (40 entries: 48 full / 64 sliding),
  `num_key_value_heads`, `head_dim`, `rms_norm_eps`,
  `max_position_embeddings`, `attention_bias`, `attention_dropout`,
  `sliding_window`, `tie_word_embeddings` (must be false), `gating`
  ("per-head" — required by the frozen-invariant check).
- `layer_types`: 40 entries of `full_attention` / `sliding_attention`
  (full at 0, 4, 8, ..., 36).
- `mlp_layer_types`: 40 entries of `dense` / `sparse` (dense only at 0). If
  omitted, derived from `mlp_only_layers` (default `[0]`) and
  `decoder_sparse_step` (default 1).
- MoE: `num_experts` 256, `num_experts_per_tok` 8, `moe_intermediate_size`
  512, `shared_expert_intermediate_size` 512, `moe_routed_scaling_factor`
  2.5, `norm_topk_prob` true, `moe_router_logit_softcapping` 0.
- `rope_parameters.sliding_attention`: `{rope_type: "default", rope_theta:
  10000.0, partial_rotary_factor: 1.0}`.
- `rope_parameters.full_attention`: `{rope_type: "yarn", rope_theta:
  500000.0, factor: 32.0, original_max_position_embeddings: 8192,
  beta_fast: 64.0, beta_slow: 1.0, partial_rotary_factor: 0.5}`. (The source
  config's `attention_factor: 1.0` is ignored — the vendored Swift
  `YarnRoPE` computes its attention scaling internally and is the pinned
  correctness reference.)
- `quantization` (or `quantization_config`): global
  `{group_size: 64, bits: 4, mode: "affine"}` plus the 39 router-gate 8-bit
  overrides listed above. The runtime validates every quantized tensor's
  stored geometry against this spec, so the overrides must match the packed
  shapes exactly.

Frozen invariants enforced at parse time (`validateFrozenInvariants`): vocab
100352, hidden 2048, intermediate 8192, 40 layers, 8 KV heads, head_dim 128,
sliding window 512, 256 experts, top-8, moe/shared intermediate 512, untied
head, per-head gating.
