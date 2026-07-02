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
    compact_kv_cache,
    gather_from_paged_cache,
    select_per_head,
)

DTYPES = [torch.bfloat16, torch.float16]
NUM_KV_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16, 32]
COMPRESSION_RATIOS = [0.25, 0.5]
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("compression_ratio", COMPRESSION_RATIOS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_compact_kv_cache(
    kv_cache_factory,
    dtype: torch.dtype,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    compression_ratio: float,
    seed: int,
) -> None:
    """Scatter tokens into the cache, compact with per-head selection,
    then gather back and verify the compacted data is correct."""
    set_random_seed(seed)
    device = "cuda"
    torch.set_default_device(device)

    seq_len = 137
    compacted_len = int(seq_len * (1 - compression_ratio))
    num_blocks = (seq_len + block_size - 1) // block_size + 4

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

    # 1. Generate dense keys and values for the sequence.
    keys_original = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype)
    values_original = torch.randn(
        seq_len, num_kv_heads, head_size, dtype=dtype,
    )

    num_seq_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_seq_blocks, dtype=torch.long, device=device)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    slot_mapping = build_slot_mapping_for_positions(
        block_table, positions, block_size,
    )

    # 2. Scatter them to the KV cache.
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    ops.reshape_and_cache(
        keys_original, values_original,
        key_cache, value_cache,
        slot_mapping, "auto", k_scale, v_scale,
    )

    # Build per-head kept_indices: each head keeps a different random
    # subset of compacted_len tokens, simulating key-diff scoring.
    kept_indices = torch.stack([
        torch.randperm(seq_len, device=device)[:compacted_len].sort().values
        for _ in range(num_kv_heads)
    ])  # [num_kv_heads, compacted_len]

    # Compute expected compacted keys/values from the original dense data.
    expected_keys = select_per_head(keys_original, kept_indices)
    expected_values = select_per_head(values_original, kept_indices)

    # 3-5. Run compact_kv_cache.
    result_len = compact_kv_cache(
        key_cache, value_cache,
        slot_mapping, kept_indices, block_table,
        block_size, num_kv_heads, head_size,
        kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale,
    )
    assert result_len == compacted_len

    # 6. Gather the compacted tokens and verify.
    compact_positions = torch.arange(
        compacted_len, dtype=torch.long, device=device,
    )
    compact_slot_mapping = build_slot_mapping_for_positions(
        block_table, compact_positions, block_size,
    )
    keys_after, values_after = gather_from_paged_cache(
        key_cache, value_cache, compact_slot_mapping,
        num_kv_heads, head_size, block_size,
    )

    torch.testing.assert_close(keys_after, expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(values_after, expected_values, atol=0, rtol=0)
