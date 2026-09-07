"""Opt-in NVFP4 PLE row gather. Host storage, GPU dequantization only on lookup.

Checkpoint layout: low nibble first, E2M1; row-major E4M3 scales per 16
columns; FP32 global scale. No scale swizzle or inverse scales.
"""
import torch
import triton
import triton.language as tl


class PackedPLEStorage:
    """Retain row-major checkpoint bytes; never expand a checkpoint shard."""

    def __init__(self, rows, dim, *, pin_memory=True):
        if rows <= 0 or dim <= 0 or dim % 16:
            raise ValueError('packed PLE needs positive rows and dimension divisible by 16')
        self.weight = torch.empty((rows, dim // 2), dtype=torch.uint8,
                                  device='cpu', pin_memory=pin_memory)
        self.scales = torch.empty((rows, dim // 16), dtype=torch.float8_e4m3fn,
                                  device='cpu', pin_memory=pin_memory)
        self.global_scale = None
        self._ranges = []
        self._tp_bounds = None
        self._source_ranges = []

    def validate_tp2_partition(self, emb):
        """Validate the actual VocabParallelEmbedding padded rank interval.

        Rank is encoded by padded bounds (the source embedding has no tp_rank
        attribute). No added vocabulary or empty rank is supported here.
        """
        indices = emb.shard_indices
        padded = emb.org_vocab_size_padded
        capacity = self.weight.shape[0]
        start = indices.padded_org_vocab_start_index
        end = indices.padded_org_vocab_end_index
        if (emb.tp_size != 2 or emb.num_added_embeddings != 0
                or padded < emb.org_vocab_size or padded % 2
                or emb.num_embeddings_padded != padded
                or capacity != padded // 2
                or emb.num_embeddings_per_partition != capacity
                or start not in (0, capacity) or end != start + capacity
                or indices.org_vocab_start_index != min(start, emb.org_vocab_size)
                or indices.org_vocab_end_index != min(end, emb.org_vocab_size)
                or indices.org_vocab_start_index >= indices.org_vocab_end_index):
            raise ValueError('invalid packed PLE TP2 padded rank layout')

    @property
    def nbytes(self):
        return self.weight.numel() + self.scales.numel()

    def load(self, packed, scales, shard_start, tp_start, tp_end, global_scale):
        import math
        if (packed.dtype != torch.uint8 or packed.ndim != 2
                or packed.shape[1] != self.weight.shape[1]
                or scales.dtype != torch.float8_e4m3fn
                or tuple(scales.shape) != (packed.shape[0], self.scales.shape[1])):
            raise ValueError('invalid packed PLE checkpoint layout or dtype')
        if (not math.isfinite(global_scale) or global_scale <= 0
                or (self.global_scale is not None and self.global_scale != global_scale)):
            raise ValueError('invalid or conflicting packed PLE global scale')
        if not (0 <= tp_start < tp_end and tp_end-tp_start <= self.weight.shape[0]
                and shard_start >= 0) or (self._tp_bounds is not None
                and self._tp_bounds != (tp_start, tp_end)):
            raise ValueError('invalid packed PLE shard bounds')
        self._tp_bounds = (tp_start, tp_end)
        self.global_scale = global_scale
        self._source_ranges.append((shard_start, shard_start + packed.shape[0]))
        start = max(shard_start, tp_start)
        end = min(shard_start + packed.shape[0], tp_end)
        if start < end:
            source = slice(start - shard_start, end - shard_start)
            target = slice(start - tp_start, end - tp_start)
            self.weight[target].copy_(packed[source])
            self.scales[target].copy_(scales[source])
            self._ranges.append((start, end))

    def finalize(self, tp_start, tp_end, *, expected_shards=None,
                 expected_global_shards=None, global_rows=None):
        if expected_global_shards is not None:
            if len(self._source_ranges) != expected_global_shards:
                raise ValueError('invalid packed PLE global shard count')
            cursor = 0
            for start, end in sorted(self._source_ranges):
                if start != cursor or end <= start:
                    raise ValueError('packed PLE has missing or overlapping global source rows')
                cursor = end
            if cursor != global_rows:
                raise ValueError('packed PLE global source is incomplete')
            shard_size = (global_rows + expected_global_shards - 1) // expected_global_shards
            expected_shards = sum(
                max(i * shard_size, tp_start) < min((i + 1) * shard_size, global_rows, tp_end)
                for i in range(expected_global_shards))
        if self._tp_bounds != (tp_start, tp_end):
            raise ValueError('invalid packed PLE finalize bounds or unloaded table')
        if expected_shards is not None and len(self._ranges) != expected_shards:
            raise ValueError('invalid packed PLE shard count')
        cursor = tp_start
        for start, end in sorted(self._ranges):
            if start != cursor:
                raise ValueError('packed PLE has missing or overlapping rows')
            cursor = end
        if cursor != tp_end:
            raise ValueError('packed PLE is incomplete')


@triton.jit
def gather_packed_kernel(
    weight_ptr, scale_ptr, ids_ptr, output_ptr, global_scale,
    embedding_dim: tl.constexpr, tp_vocab_start, tp_vocab_end,
    fp8_reference: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids_ptr + row).to(tl.int64)
    valid = (index >= tp_vocab_start) & (index < tp_vocab_end)
    local = tl.where(valid, index - tp_vocab_start, 0).to(tl.int64)
    col = tl.arange(0, BLOCK_D)
    mask = col < embedding_dim
    weights = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    scales = scale_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    packed = tl.load(weights + local * (embedding_dim // 2) + col // 2,
                     mask=valid & mask, other=0)
    nibble = tl.where((col & 1) == 0, packed & 15, packed >> 4)
    code = nibble & 7
    magnitude = tl.where(code == 0, 0., tl.where(code == 1, .5,
                tl.where(code == 2, 1., tl.where(code == 3, 1.5,
                tl.where(code == 4, 2., tl.where(code == 5, 3.,
                tl.where(code == 6, 4., 6.)))))))
    quantized = tl.where((nibble & 8) != 0, -magnitude, magnitude)
    scale = tl.load(scales + local * (embedding_dim // 16) + col // 16,
                    mask=valid & mask, other=0.).to(tl.float32)
    if fp8_reference:
        # Match original SGLang loader multiplication order and FP8 storage
        # rounding, then the existing pinned gather's BF16 conversion.
        values = quantized * (scale * global_scale)
        bits = values.to(tl.int32, bitcast=True)
        # E4M3 RNE explicitly, including subnormals. Triton 3.7's interpreter
        # mishandles subnormal downcasts; using exact power-of-two steps also
        # makes the reference path independent of GPU satfinite conversion.
        step_bits = tl.maximum(((bits >> 23) & 255) - 3, 118) << 23
        step = step_bits.to(tl.float32, bitcast=True)
        units = tl.abs(values) / step
        lower = units.to(tl.int32)
        fraction = units - lower
        rounded = (lower + ((fraction > .5) | ((fraction == .5) & ((lower & 1) != 0))).to(tl.int32)) * step
        # This image's PyTorch float8_e4m3fn cast saturates finite overflow.
        rounded = tl.where(values != values, float("nan"), tl.minimum(rounded, 448.))
        # LLVM's no-signed-zeros algebra can erase E2M1 -0 during multiplies.
        # Restore the checkpoint sign explicitly for bitwise legacy agreement.
        sign = ((nibble.to(tl.int32) & 8) << 28) ^ (scale.to(tl.int32, bitcast=True) & -2147483648)
        values = (rounded.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    else:
        # Explicit BF16 RNE (also avoids Triton interpreter's BF16 truncation).
        values = quantized * scale * global_scale
        bits = values.to(tl.int32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
        values = bits.to(tl.float32, bitcast=True)
    tl.store(output_ptr + row.to(tl.int64) * embedding_dim + col,
             tl.where(valid, values, 0.), mask=mask)
