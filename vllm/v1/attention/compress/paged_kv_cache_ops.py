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
