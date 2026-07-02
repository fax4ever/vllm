# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Operations for gathering from and scattering to the ROCm paged KV cache.

The ROCm paged KV cache uses a 5D key layout and a 4D value layout:
  - key_cache:   [num_blocks, num_kv_heads, head_size // x, block_size, x]
  - value_cache: [num_blocks, num_kv_heads, head_size, block_size]

where x = 16 // element_size is the packed inner dimension for keys.

These helpers convert between this paged layout and the dense
[num_tokens, num_kv_heads, head_size] format used by reshape_and_cache
and the model's K/V projections.
"""

import torch

from vllm import _custom_ops as ops


def build_slot_mapping_for_positions(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build a slot mapping for given positions using a block table.

    Each slot identifies the cell in the paged cache dedicated to storing
    a token's key and value data across all KV heads. The same block_idx
    and offset address the corresponding entries in both the 5D key cache
    and the 4D value cache.

    Args:
        block_table: [max_num_blocks_per_seq] mapping logical block index
            to physical block index for a single sequence.
        positions: [num_positions] token positions in the sequence.
        block_size: number of slots per physical block.

    Returns:
        slot_mapping: [num_positions] flat slot indices, where each slot
            is decomposed as block_idx * block_size + offset.
    """
    logical_block_indices = positions // block_size
    physical_block_indices = block_table[logical_block_indices]
    offsets = positions % block_size
    return physical_block_indices * block_size + offsets


def gather_from_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather dense K/V tensors from the ROCm paged cache.

    Reads tokens from the 5D key cache and 4D value cache into the
    dense [num_tokens, num_kv_heads, head_size] format that
    reshape_and_cache expects as input.

    Args:
        key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        slot_mapping: [num_tokens] flat slot indices.
        num_kv_heads: number of KV heads.
        head_size: dimension per head.
        block_size: slots per block.

    Returns:
        keys: [num_tokens, num_kv_heads, head_size] dense keys.
        values: [num_tokens, num_kv_heads, head_size] dense values.
    """
    block_indices = slot_mapping // block_size
    offsets = slot_mapping % block_size

    keys_packed = key_cache[block_indices, :, :, offsets, :]
    keys = keys_packed.reshape(-1, num_kv_heads, head_size)

    values = value_cache[block_indices, :, :, offsets]

    return keys, values


def select_per_head(
    dense: torch.Tensor,
    kept_indices: torch.Tensor,
) -> torch.Tensor:
    """Select tokens per head from a dense tensor.

    Args:
        dense: [seq_len, num_kv_heads, head_size]
        kept_indices: [num_kv_heads, compacted_len] — which token
            positions to keep, independently per head.

    Returns:
        compacted: [compacted_len, num_kv_heads, head_size]
    """
    head_size = dense.shape[2]
    by_head = dense.permute(1, 0, 2)  # [num_kv_heads, seq_len, head_size]
    idx = kept_indices.unsqueeze(-1).expand(-1, -1, head_size)
    kept = by_head.gather(1, idx)  # [num_kv_heads, compacted_len, head_size]
    return kept.permute(1, 0, 2).contiguous()


def compact_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kept_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> int:
    """Compact a single sequence's KV cache using per-head token selection.

    Gathers the full cached context into dense form, selects the kept
    tokens per head, and scatters the compacted result back into the
    first compacted_len slots of the paged cache.

    Keys and values are gathered separately so only one full dense buffer
    is in memory at a time.

    Args:
        key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        slot_mapping: [seq_len] flat slot indices for the sequence's
            cached positions.
        kept_indices: [num_kv_heads, compacted_len] — which token
            positions to keep, independently per head.
        block_table: [max_num_blocks_per_seq] for this sequence.
        block_size: slots per block.
        num_kv_heads: number of KV heads.
        head_size: dimension per head.
        kv_cache_dtype: cache dtype string for reshape_and_cache.
        k_scale: key quantization scale.
        v_scale: value quantization scale.

    Returns:
        compacted_len: number of tokens after compaction.
    """
    device = key_cache.device
    compacted_len = kept_indices.shape[1]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Step 1: Gather keys and select kept tokens.
    keys_dense, _ = gather_from_paged_cache(
        key_cache, value_cache, slot_mapping,
        num_kv_heads, head_size, block_size,
    )
    keys_compact = select_per_head(keys_dense, kept_indices)
    del keys_dense

    # Step 2: Gather values and select kept tokens.
    _, values_dense = gather_from_paged_cache(
        key_cache, value_cache, slot_mapping,
        num_kv_heads, head_size, block_size,
    )
    values_compact = select_per_head(values_dense, kept_indices)
    del values_dense

    # Step 3: Scatter compacted tokens back into slots 0..compacted_len-1.
    compact_positions = torch.arange(
        compacted_len, dtype=torch.long, device=device,
    )
    compact_slot_mapping = build_slot_mapping_for_positions(
        block_table, compact_positions, block_size,
    )
    ops.reshape_and_cache(
        keys_compact, values_compact,
        key_cache, value_cache,
        compact_slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )

    return compacted_len
