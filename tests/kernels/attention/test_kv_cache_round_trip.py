# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Round-trip validation for paged KV cache gather and scatter.

Verifies that reading tokens from the ROCm 5D/4D paged cache into dense
tensors and writing them back with reshape_and_cache preserves the cache
contents exactly. This is the foundation for KV cache compaction: if the
round-trip is lossless, inserting per-head selection between gather and
scatter should be safe.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.compress.paged_kv_cache_ops import (
    build_slot_mapping_for_positions,
    gather_from_paged_cache,
)

DTYPES = [torch.bfloat16, torch.float16]
NUM_KV_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16, 32]
SEEDS = [0]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_kv_cache_round_trip(
    kv_cache_factory,
    dtype: torch.dtype,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
) -> None:
    """Gather all tokens from the paged cache, scatter them back, and verify
    the cache is unchanged."""
    set_random_seed(seed)
    device = "cuda"
    torch.set_default_device(device)

    seq_len = 137
    num_blocks = (seq_len + block_size - 1) // block_size + 4  # extra blocks

    key_caches, value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,  # num_layers
        num_kv_heads,
        head_size,
        "auto",
        dtype,
        seed,
        device,
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]

    # Write known data into the cache for a single sequence.
    keys_original = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype)
    values_original = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype)

    # Build a block table: assign physical blocks sequentially.
    num_seq_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_seq_blocks, dtype=torch.long, device=device)

    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    slot_mapping = build_slot_mapping_for_positions(
        block_table, positions, block_size
    )

    # Scatter original data into the cache.
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    ops.reshape_and_cache(
        keys_original,
        values_original,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        k_scale,
        v_scale,
    )

    # Snapshot the cache after the initial write.
    key_cache_snapshot = key_cache.clone()
    value_cache_snapshot = value_cache.clone()

    # --- Round trip: gather then scatter back ---
    keys_dense, values_dense = gather_from_paged_cache(
        key_cache, value_cache, slot_mapping,
        num_kv_heads, head_size, block_size,
    )

    # Verify the gathered data matches what we originally wrote.
    torch.testing.assert_close(keys_dense, keys_original, atol=0, rtol=0)
    torch.testing.assert_close(values_dense, values_original, atol=0, rtol=0)

    # Scatter the gathered data back into the cache.
    ops.reshape_and_cache(
        keys_dense,
        values_dense,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        k_scale,
        v_scale,
    )

    # Verify the cache is unchanged after the round trip.
    torch.testing.assert_close(key_cache, key_cache_snapshot, atol=0, rtol=0)
    torch.testing.assert_close(
        value_cache, value_cache_snapshot, atol=0, rtol=0
    )
