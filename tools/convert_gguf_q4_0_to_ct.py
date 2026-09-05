#!/usr/bin/env python3
"""
Convert a Q4_0 GGUF quantized model to compressed-tensors "pack-quantized" format.

The Q4_0 blocks in GGUF and CT "pack-quantized" groups both quantize along the
same feature dimension, so the conversion is lossless at the nibble level --
quantized weight values and scales are preserved exactly.

GGUF data layout (from GGUFReader):
  - tensor.shape = [ne0, ne1, ...]  (GGML order, ne0=innermost)
  - tensor.data shape: for 2D → (ne1, ne0*18/32); for 3D → (ne2, ne1, ne0*18/32)
  - Q4_0 blocks run along ne0

CT pack-quantized format:
  - weight_packed: [out_features, ceil(in_features / 8)]
  - weight_scale:  [out_features, ceil(in_features / group_size)]
  - weight_shape:  [out_features, in_features]

Usage:
    python tools/convert_gguf_q4_0_to_ct.py \
        --gguf-path models/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
        --w4a16-ref models/gemma-4-26B-A4B-it-W4A16 \
        --output-dir models/gemma-4-26B-A4B-it-ct-q4_0
"""

import argparse
import json
import os
import re
from typing import Optional

import numpy as np
import safetensors.torch as st
import torch
from gguf import GGMLQuantizationType, GGUFReader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QK4_0 = 32
Q4_0_BLOCK_BYTES = 18  # 2 bytes fp16 scale + 16 bytes packed uint4
PACK_FACTOR = 32 // 4  # 8 int4 values per int32
INTS_PER_BLOCK = QK4_0 // PACK_FACTOR  # 4 int32 per Q4_0 block
MARLIN_MIN_K = 128  # Marlin kernel requires in_features % min_thread_k == 0

# Padded weight zero-fill nibble: q_signed=0 → q_unsigned=8 → nibble 0x8
# Each int32 packs 8 nibbles of 0x8 → 0x88888888
_ZERO_PACKED_INT32 = np.array([0x88888888], dtype=np.uint32).view(np.int32)[0]

# ---------------------------------------------------------------------------
# Q4_0 block → CT pack-quantized conversion (lossless)
# ---------------------------------------------------------------------------


def _convert_blocks_to_ct(raw_data: np.ndarray, ne0: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Q4_0 block data to CT pack-quantized format (vectorized).

    Args:
        raw_data: uint8 numpy array of shape (*batch_dims, ne0 * 18 // 32)
                  Each "row" along the last axis contains ne0/32 Q4_0 blocks
                  concatenated as raw bytes.
        ne0:      Number of values in the quantized dimension (must be divisible by 32).

    Returns:
        scales: float16 array of shape (*batch_dims, ne0 // 32)
        packed: int32 array of shape (*batch_dims, ne0 // 8)
    """
    assert ne0 % QK4_0 == 0, f"ne0={ne0} not divisible by {QK4_0}"
    blocks_per_row = ne0 // QK4_0
    expected_last_dim = blocks_per_row * Q4_0_BLOCK_BYTES
    assert raw_data.shape[-1] == expected_last_dim, \
        f"Expected last dim {expected_last_dim}, got {raw_data.shape[-1]}"

    batch_shape = raw_data.shape[:-1]
    n_rows = int(np.prod(batch_shape))
    total_blocks = n_rows * blocks_per_row

    # Reshape to (n_rows, blocks_per_row, 18)
    flat = raw_data.reshape(n_rows, blocks_per_row, Q4_0_BLOCK_BYTES)

    # ── Scales: first 2 bytes of each block as fp16 ──
    scale_bytes_all = flat[:, :, :2].reshape(-1, 2)  # (total_blocks, 2)
    scales_all = np.frombuffer(scale_bytes_all.tobytes(), dtype=np.float16)
    scales = scales_all.reshape(n_rows, blocks_per_row).copy()

    # ── Q values: bytes 2..17, unpack nibbles ──
    q_bytes_all = flat[:, :, 2:]  # (n_rows, blocks_per_row, 16)
    # Low nibble: q[0..15], High nibble: q[16..31]
    q_low = (q_bytes_all & 0x0F).astype(np.uint8)    # (n_rows, blocks_per_row, 16)
    q_high = (q_bytes_all >> 4).astype(np.uint8)     # (n_rows, blocks_per_row, 16)
    # Concatenate to get 32 values per block
    q_all = np.concatenate([q_low, q_high], axis=2)  # (n_rows, blocks_per_row, 32)

    # ── Pack into CT int32 format: 8 values per int32 ──
    # q_all shape: (n_rows, blocks_per_row, 32) → (n_rows, blocks_per_row, 4, 8)
    q_groups = q_all.reshape(n_rows, blocks_per_row, INTS_PER_BLOCK, PACK_FACTOR)

    # Pack: v0 | v1<<4 | v2<<8 | ... | v7<<28
    # Use uint32 to avoid NumPy's overflow-to-int64 promotion, then view as int32.
    q_groups_u32 = q_groups.astype(np.uint32)
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)  # (8,)
    packed_u32 = (q_groups_u32 << shifts[np.newaxis, np.newaxis, np.newaxis, :]).sum(
        axis=3, dtype=np.uint32)
    packed = packed_u32.view(np.int32)
    # packed shape: (n_rows, blocks_per_row, 4)

    packed = packed.reshape(n_rows, blocks_per_row * INTS_PER_BLOCK)
    packed = packed.reshape(*batch_shape, blocks_per_row * INTS_PER_BLOCK)

    return scales, packed


# ---------------------------------------------------------------------------
# Marlin kernel in_features padding
# ---------------------------------------------------------------------------


def _pad_ct_to_marlin(scales: np.ndarray, packed: np.ndarray,
                      in_features: int, min_k: int = MARLIN_MIN_K
                      ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pad CT pack-quantized tensors so in_features is divisible by min_k.

    Padded groups are zero-fill — dequantized weight = 0 regardless of scale.

    Returns:
        scales_padded, packed_padded, padded_in_features
    """
    if in_features % min_k == 0:
        return scales, packed, in_features

    padded_in = ((in_features + min_k - 1) // min_k) * min_k
    pad_vals = padded_in - in_features
    pad_groups = pad_vals // QK4_0
    assert pad_vals % QK4_0 == 0, f"pad_vals={pad_vals} not multiple of {QK4_0}"

    # Pad scales: zero-fill (dequantized weight = 0 regardless of scale)
    pad_dtype = scales.dtype
    if scales.ndim >= 2:
        scales_pad = np.zeros(scales.shape[:-1] + (pad_groups,), dtype=pad_dtype)
        scales_out = np.concatenate([scales, scales_pad], axis=-1)
    else:
        scales_out = np.concatenate([scales, np.zeros(pad_groups, dtype=pad_dtype)])

    # Pad packed: each padded group → 4 int32s of 0x88888888
    pad_ints = pad_groups * INTS_PER_BLOCK
    if packed.ndim >= 2:
        packed_pad = np.full(packed.shape[:-1] + (pad_ints,), _ZERO_PACKED_INT32,
                             dtype=np.int32)
        packed_out = np.concatenate([packed, packed_pad], axis=-1)
    else:
        packed_out = np.concatenate([packed,
                                     np.full(pad_ints, _ZERO_PACKED_INT32, dtype=np.int32)])

    return scales_out, packed_out, padded_in


# ---------------------------------------------------------------------------
# Tensor name mapping: GGUF → HuggingFace / compressed-tensors
# ---------------------------------------------------------------------------


def _maybe_transpose_for_hf(arr: np.ndarray, gguf_shape: list[int]) -> np.ndarray:
    """
    GGUF stores weight matrices as [in_features, out_features] (GGML layout).
    HuggingFace stores them as [out_features, in_features].
    Transpose 2D tensors; leave 1D tensors (norms, scales) unchanged.
    """
    if arr.ndim == 2:
        return arr.T
    return arr


def _map_block_tensor(gguf_name: str, moe_registry: dict) -> Optional[str]:
    """Map a blk.* GGUF tensor name to an HF-equivalent base name."""
    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return None
    bid = int(m.group(1))
    rest = m.group(2)
    base = f"model.language_model.layers.{bid}"

    # Attention projections (Q4_0 quantized)
    if rest in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
        proj = rest[5]
        return f"{base}.self_attn.{proj}_proj.weight"
    elif rest == "attn_output.weight":
        return f"{base}.self_attn.o_proj.weight"

    # Regular MLP (Q4_0 quantized)
    elif rest == "ffn_gate.weight":
        return f"{base}.mlp.gate_proj.weight"
    elif rest == "ffn_up.weight":
        return f"{base}.mlp.up_proj.weight"
    elif rest == "ffn_down.weight":
        return f"{base}.mlp.down_proj.weight"

    # Norms (unquantized)
    elif rest == "attn_q_norm.weight":
        return f"{base}.self_attn.q_norm.weight"
    elif rest == "attn_k_norm.weight":
        return f"{base}.self_attn.k_norm.weight"
    elif rest == "attn_norm.weight":
        return f"{base}.input_layernorm.weight"
    elif rest == "ffn_norm.weight":
        return f"{base}.pre_feedforward_layernorm.weight"
    elif rest == "post_attention_norm.weight":
        return f"{base}.post_attention_layernorm.weight"
    elif rest == "post_ffw_norm.weight":
        return f"{base}.post_feedforward_layernorm.weight"
    elif rest == "post_ffw_norm_1.weight":
        return f"{base}.post_feedforward_layernorm_1.weight"
    elif rest == "post_ffw_norm_2.weight":
        return f"{base}.post_feedforward_layernorm_2.weight"
    elif rest == "pre_ffw_norm_2.weight":
        return f"{base}.pre_feedforward_layernorm_2.weight"

    # Router components (unquantized)
    elif rest == "ffn_gate_inp.weight":
        return f"{base}.router.proj.weight"
    elif rest == "ffn_gate_inp.scale":
        return f"{base}.router.scale"
    elif rest == "ffn_down_exps.scale":
        return f"{base}.router.per_expert_scale"

    # Layer output scale
    elif rest == "layer_output_scale.weight":
        return f"{base}.layer_scalar"

    # MoE stacked weights (Q4_0, 3D → split by expert)
    elif rest == "ffn_down_exps.weight":
        moe_registry[gguf_name] = {"type": "down", "bid": bid}
        return None
    elif rest == "ffn_gate_up_exps.weight":
        moe_registry[gguf_name] = {"type": "gate_up", "bid": bid}
        return None

    return None


_QUANTIZED_HF_SUFFIXES = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
]


def _is_quantized_weight(hf_name: str) -> bool:
    return any(hf_name.endswith(s) for s in _QUANTIZED_HF_SUFFIXES)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Q4_0 GGUF to compressed-tensors format"
    )
    p.add_argument("--gguf-path", required=True)
    p.add_argument("--w4a16-ref", required=True,
                   help="Reference W4A16 model dir (config + vision tower)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   help="Dtype for scales and unquantized weights (default: bfloat16)")
    p.add_argument("--group-size", type=int, default=32,
                   help="Group size for config (default: 32, must match Q4_0 block size)")
    p.add_argument("--no-marlin-pad", action="store_true",
                   help="Skip padding weights for Marlin K constraint")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (
        torch.float16 if args.dtype == "float16" else torch.float32)

    # ── Load GGUF ──────────────────────────────────────────────────────────
    print(f"[load] Reading GGUF: {args.gguf_path}")
    reader = GGUFReader(args.gguf_path)
    gguf_info = {}
    for t in reader.tensors:
        gguf_info[t.name] = {
            "shape": [int(d) for d in t.shape],
            "type": int(t.tensor_type),
            "n_elements": int(t.n_elements),
            "n_bytes": int(t.n_bytes),
            "data": np.asarray(t.data),
        }

    print(f"[load] {len(gguf_info)} tensors found")
    for qn, qv in GGMLQuantizationType.__members__.items():
        c = sum(1 for t in gguf_info.values() if t["type"] == qv.value)
        if c:
            print(f"  {qn} (type={qv.value}): {c}")

    # ── Name mapping ──────────────────────────────────────────────────────
    name_map = {}
    moe_registry = {}

    for gguf_name in gguf_info:
        if gguf_name == "token_embd.weight":
            name_map[gguf_name] = "model.language_model.embed_tokens.weight"
        elif gguf_name == "output_norm.weight":
            name_map[gguf_name] = "model.language_model.norm.weight"
        elif gguf_name == "rope_freqs.weight":
            continue  # not needed in HF
        elif gguf_name.startswith("blk."):
            hf = _map_block_tensor(gguf_name, moe_registry)
            if hf:
                name_map[gguf_name] = hf

    print(f"[map] {len(name_map)} GGUF → HF name mappings")
    print(f"[moe] {len(moe_registry)} MoE stacked tensors to split")

    # ── Load reference config ─────────────────────────────────────────────
    with open(os.path.join(args.w4a16_ref, "config.json")) as f:
        config = json.load(f)

    num_experts = config.get("text_config", {}).get("num_experts", 128)
    num_layers = config.get("text_config", {}).get("num_hidden_layers", 30)

    # ── Copy non-weight files from reference ──────────────────────────────
    import shutil
    for fn in ["tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "processor_config.json",
               "chat_template.jinja"]:
        src = os.path.join(args.w4a16_ref, fn)
        dst = os.path.join(args.output_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # ── Collect output tensors ────────────────────────────────────────────
    ct_tensors: dict[str, torch.Tensor] = {}
    stats = {"q4_0_2d": 0, "q4_0_3d_experts": 0, "passthrough": 0}

    # --- Passthrough: non-quantized tensors ---
    for gguf_name, info in tqdm(list(gguf_info.items()), desc="Passthrough"):
        if gguf_name not in name_map:
            continue
        hf_name = name_map[gguf_name]
        if _is_quantized_weight(hf_name):
            continue  # handled below

        data = info["data"]
        qtype = info["type"]
        # HF target shape: (out_features, in_features) = (ne1, ne0) for 2D,
        # matching GGUFReader's decoded data layout for F32.
        hf_shape = tuple(reversed(info["shape"])) if len(info["shape"]) >= 2 else tuple(info["shape"])

        if qtype == GGMLQuantizationType.F32.value:
            # GGUFReader already decodes F32 to float32 in HF order.
            ct_tensors[hf_name] = torch.from_numpy(
                data.astype(np.float32, copy=True)).to(torch_dtype)
        elif qtype == GGMLQuantizationType.F16.value:
            arr = np.frombuffer(data.ravel().tobytes()[:info["n_elements"] * 2],
                                dtype=np.float16).reshape(hf_shape)
            ct_tensors[hf_name] = torch.from_numpy(arr.copy()).to(torch_dtype)
        elif qtype == GGMLQuantizationType.BF16.value:
            arr = np.frombuffer(data.ravel().tobytes()[:info["n_elements"] * 2],
                                dtype=np.uint16).reshape(hf_shape)
            ct_tensors[hf_name] = torch.from_numpy(
                arr.view(np.float32).copy()).to(torch_dtype)
        else:
            # Other quant types (e.g. Q6_K for embeddings) → dequantize
            from gguf import quants as q
            tensor_type = GGMLQuantizationType(qtype)
            dq = q.dequantize(data.ravel()[:info["n_bytes"]], tensor_type)
            dq = dq.reshape(hf_shape)
            ct_tensors[hf_name] = torch.from_numpy(dq.copy()).to(torch_dtype)

        stats["passthrough"] += 1

    # --- Q4_0 2D weights: lossless nibble conversion ---
    for gguf_name, info in tqdm(list(gguf_info.items()), desc="Q4_0 2D"):
        if info["type"] != GGMLQuantizationType.Q4_0.value:
            continue
        if gguf_name not in name_map:
            continue
        hf_name = name_map[gguf_name]
        if not _is_quantized_weight(hf_name):
            continue

        shape = info["shape"]
        if len(shape) != 2:
            continue  # 3D MoE handled below
        ne0, ne1 = shape  # GGML: ne0=innermost (in_features), ne1=out_features

        if ne0 % QK4_0 != 0:
            print(f"[warn] {gguf_name}: ne0={ne0} not divisible by {QK4_0}, skipping")
            continue

        raw = info["data"]  # shape: (ne1, ne0*18/32)
        scales, packed = _convert_blocks_to_ct(raw, ne0)
        # CT: weight_shape = [out_features=ne1, in_features=ne0]
        ct_tensors[f"{hf_name}_packed"] = torch.from_numpy(packed)
        ct_tensors[f"{hf_name}_scale"] = torch.from_numpy(scales).to(torch_dtype)
        ct_tensors[f"{hf_name}_shape"] = torch.tensor([ne1, ne0], dtype=torch.int64)
        stats["q4_0_2d"] += 1

    # --- Q4_0 3D MoE weights: lossless nibble conversion per expert ---
    for gguf_name, info in tqdm(list(gguf_info.items()), desc="Q4_0 3D MoE"):
        if info["type"] != GGMLQuantizationType.Q4_0.value:
            continue
        if gguf_name not in moe_registry:
            continue

        moe = moe_registry[gguf_name]
        bid = moe["bid"]
        shape = info["shape"]
        if len(shape) != 3:
            print(f"[warn] {gguf_name}: expected 3D, got shape {shape}")
            continue

        ne0, ne1, ne2 = shape  # GGML: ne0=innermost, ne2=outermost
        raw = info["data"]  # shape: (ne2, ne1, ne0*18/32)

        assert ne2 == num_experts, \
            f"Expected {num_experts} experts, got {ne2} from shape {shape}"
        assert ne0 % QK4_0 == 0, f"ne0={ne0} not divisible by {QK4_0}"

        if moe["type"] == "down":
            # ffn_down_exps: shape [ne0, ne1, ne2] = [intermediate, hidden, experts]
            # raw shape: (ne2=experts, ne1=hidden, ne0*18/32)
            # Expert e: raw[e] → [ne1, ne0*18/32] → CT weight_shape=[ne1, ne0]
            for e in range(ne2):
                expert_raw = raw[e]  # [ne1, ne0*18/32]
                scales, packed = _convert_blocks_to_ct(expert_raw, ne0)
                base = f"model.language_model.layers.{bid}.experts.{e}.down_proj.weight"
                ct_tensors[f"{base}_packed"] = torch.from_numpy(packed)
                ct_tensors[f"{base}_scale"] = torch.from_numpy(scales).to(torch_dtype)
                ct_tensors[f"{base}_shape"] = torch.tensor([ne1, ne0], dtype=torch.int64)

        elif moe["type"] == "gate_up":
            # ffn_gate_up_exps: shape [ne0, ne1, ne2] = [hidden, 2*intermediate, experts]
            # raw shape: (ne2=experts, ne1=2*intermediate, ne0*18/32)
            intermediate = ne1 // 2
            assert intermediate * 2 == ne1, f"ne1={ne1} not evenly divisible by 2"
            for e in range(ne2):
                expert_raw = raw[e]  # [ne1, ne0*18/32]
                gate_raw = expert_raw[:intermediate]
                up_raw = expert_raw[intermediate:]

                gate_s, gate_p = _convert_blocks_to_ct(gate_raw, ne0)
                up_s, up_p = _convert_blocks_to_ct(up_raw, ne0)

                base_g = f"model.language_model.layers.{bid}.experts.{e}.gate_proj.weight"
                base_u = f"model.language_model.layers.{bid}.experts.{e}.up_proj.weight"
                ct_tensors[f"{base_g}_packed"] = torch.from_numpy(gate_p)
                ct_tensors[f"{base_g}_scale"] = torch.from_numpy(gate_s).to(torch_dtype)
                ct_tensors[f"{base_g}_shape"] = torch.tensor([intermediate, ne0], dtype=torch.int64)
                ct_tensors[f"{base_u}_packed"] = torch.from_numpy(up_p)
                ct_tensors[f"{base_u}_scale"] = torch.from_numpy(up_s).to(torch_dtype)
                ct_tensors[f"{base_u}_shape"] = torch.tensor([intermediate, ne0], dtype=torch.int64)

        stats["q4_0_3d_experts"] += ne2

    # ── Copy vision tower from reference ──────────────────────────────────
    ref_dir = args.w4a16_ref
    # Check for sharded safetensors first
    index_path = os.path.join(ref_dir, "model.safetensors.index.json")
    vision_keys = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        vision_files = set()
        for k, shard in idx["weight_map"].items():
            if k.startswith("model.vision_") or k.startswith("model.embed_vision"):
                vision_keys.append(k)
                vision_files.add(os.path.join(ref_dir, shard))
        for shard_path in sorted(vision_files):
            with st.safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("model.vision_") or key.startswith("model.embed_vision"):
                        ct_tensors[key] = f.get_tensor(key)
    else:
        ref_st_path = os.path.join(ref_dir, "model.safetensors")
        if os.path.exists(ref_st_path):
            with st.safe_open(ref_st_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("model.vision_") or key.startswith("model.embed_vision"):
                        ct_tensors[key] = f.get_tensor(key)
                        vision_keys.append(key)

    n_vis = len(vision_keys)
    if n_vis:
        print(f"[vision] Copied {n_vis} vision tower tensors from reference")
    else:
        print(f"[vision] No vision tower tensors found in reference (text-only model)")

    # ── Marlin K padding ──────────────────────────────────────────────────
    # Marlin kernel requires in_features % min_thread_k == 0.
    # Only down_proj weights fail (K=704 or 2112 not divisible by 128).
    # Because down_proj input = SiLU(gate) * up, we also pad gate/up rows
    # so the intermediate dimension stays consistent across the FFN block.
    if not args.no_marlin_pad:
        padded_count = 0
        # Collect all weight_packed keys that need padding
        pad_keys = []
        for key in sorted(ct_tensors.keys()):
            if not key.endswith(".weight_packed"):
                continue
            shape_key = key.replace("_packed", "_shape")
            if shape_key not in ct_tensors:
                continue
            out_f, in_f = ct_tensors[shape_key].tolist()
            if in_f % MARLIN_MIN_K != 0 and ".down_proj." in key:
                pad_keys.append(key)

        for key in pad_keys:
            shape_key = key.replace("_packed", "_shape")
            scale_key = key.replace("_packed", "_scale")
            out_f, in_f = ct_tensors[shape_key].tolist()
            padded_in = ((in_f + MARLIN_MIN_K - 1) // MARLIN_MIN_K) * MARLIN_MIN_K
            pad_rows = padded_in - in_f  # number of extra rows/cols

            # Pad down_proj (column padding: add groups to in_features)
            scales = ct_tensors[scale_key].float().numpy()
            packed = ct_tensors[key].numpy()
            scales_p, packed_p, _ = _pad_ct_to_marlin(scales, packed, in_f)
            ct_tensors[scale_key] = torch.from_numpy(scales_p)
            ct_tensors[key] = torch.from_numpy(packed_p)
            ct_tensors[shape_key] = torch.tensor([out_f, padded_in], dtype=torch.int64)

            # Pad corresponding gate_proj and up_proj (row padding: add rows to out_features)
            # gate/up share the same intermediate dim as down_proj's in_features
            gate_key = key.replace(".down_proj.", ".gate_proj.")
            up_key = key.replace(".down_proj.", ".up_proj.")

            for companion_key in (gate_key, up_key):
                if companion_key not in ct_tensors:
                    continue
                c_shape_key = companion_key.replace("_packed", "_shape")
                c_scale_key = companion_key.replace("_packed", "_scale")
                c_out_f, c_in_f = ct_tensors[c_shape_key].tolist()

                c_scales = ct_tensors[c_scale_key].float().numpy()
                c_packed = ct_tensors[companion_key].numpy()

                # Pad out_features (rows) for gate/up
                pad_scales = np.zeros((pad_rows,) + c_scales.shape[1:], dtype=c_scales.dtype)
                c_scales_p = np.concatenate([c_scales, pad_scales], axis=0)
                pad_packed = np.full((pad_rows,) + c_packed.shape[1:],
                                     _ZERO_PACKED_INT32, dtype=np.int32)
                c_packed_p = np.concatenate([c_packed, pad_packed], axis=0)

                ct_tensors[c_scale_key] = torch.from_numpy(c_scales_p)
                ct_tensors[companion_key] = torch.from_numpy(c_packed_p)
                ct_tensors[c_shape_key] = torch.tensor(
                    [c_out_f + pad_rows, c_in_f], dtype=torch.int64)

            padded_count += 1

        if padded_count:
            # Update config with padded intermediate sizes
            text_cfg = config.setdefault("text_config", {})
            old_inter = text_cfg.get("intermediate_size", 0)
            old_moe_inter = text_cfg.get("moe_intermediate_size", 0)
            if old_inter and old_inter % MARLIN_MIN_K != 0:
                new_inter = ((old_inter + MARLIN_MIN_K - 1) // MARLIN_MIN_K) * MARLIN_MIN_K
                text_cfg["intermediate_size"] = new_inter
                print(f"[marlin] intermediate_size: {old_inter} → {new_inter}")
            if old_moe_inter and old_moe_inter % MARLIN_MIN_K != 0:
                new_moe = ((old_moe_inter + MARLIN_MIN_K - 1) // MARLIN_MIN_K) * MARLIN_MIN_K
                text_cfg["moe_intermediate_size"] = new_moe
                print(f"[marlin] moe_intermediate_size: {old_moe_inter} → {new_moe}")
            print(f"[marlin] Padded {padded_count} FFN blocks for Marlin compatibility")

    # ── Update config.json ────────────────────────────────────────────────
    qc = config.setdefault("quantization_config", {})
    qc["quant_method"] = "compressed-tensors"
    qc["format"] = "pack-quantized"
    qc["quantization_status"] = "compressed"
    qc["global_compression_ratio"] = None
    qc["ignore"] = [
        "lm_head",
        *[f"model.language_model.layers.{i}.router.proj" for i in range(num_layers)],
        "re:.*embed.*",
        "re:.*vision.*",
        "re:.*audio.*",
    ]
    qc["config_groups"] = {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "group_size": args.group_size,
                "num_bits": 4,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        }
    }
    config["dtype"] = args.dtype

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Save safetensors ──────────────────────────────────────────────────
    print(f"[save] Writing {len(ct_tensors)} tensors to model.safetensors ...")
    st.save_file(ct_tensors, os.path.join(args.output_dir, "model.safetensors"))

    # ── Stats ─────────────────────────────────────────────────────────────
    print(f"\n[stats]")
    print(f"  Q4_0 2D (lossless):         {stats['q4_0_2d']} tensors")
    print(f"  Q4_0 3D expert weights:     {stats['q4_0_3d_experts']}")
    print(f"  Passthrough:                {stats['passthrough']}")
    print(f"  Total output tensors:       {len(ct_tensors)}")


if __name__ == "__main__":
    main()
