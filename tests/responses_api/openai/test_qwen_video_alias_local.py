"""CPU regression for release-name video metadata, timestamps and image parity."""
import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import utils  # Bootstrap CPU-only serving imports before processor imports.
import pytest
import torch
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.multimodal.processors import qwen_vl as qv


class Tokenizer:
    def encode(self, text, **kwargs):
        return [300 + ord(char) for char in text]


def processor(model_type):
    snapshot = Path(os.environ['QWEN_REPLAY_MODEL_PATH'])
    config = Qwen4ExpConfig(**json.loads((snapshot / 'config.json').read_text()))
    assert config.model_type == 'qwen3_8_flash_next'
    config.model_type = model_type
    def initialize_base(instance, hf_config, *args, **kwargs):
        instance.hf_config = hf_config
        instance._processor = SimpleNamespace(tokenizer=Tokenizer())
    with patch.object(qv.SGLangBaseProcessor, '__init__', initialize_base), patch.object(
        qv.MultimodalSpecialTokens, 'build', lambda instance, _: instance
    ):
        return qv.QwenVLImageProcessor(config, SimpleNamespace(), SimpleNamespace())


@pytest.mark.parametrize('model_type', ['qwen4_exp', 'qwen3_8_flash_next'])
def test_release_processor_enables_existing_worker_policy(model_type):
    instance = processor(model_type)
    assert instance.supports_mm_processor_concurrency
    assert instance.auto_mm_processor_worker_num == 2
    assert instance.auto_mm_io_worker_num == 16


@pytest.mark.parametrize('model_type', ['qwen4_exp', 'qwen3_8_flash_next', 'qwen2_vl'])
def test_preprocessed_video_preserves_metadata_and_sampling(model_type):
    instance = processor(model_type)
    instance.video_config = {'fps': 2}
    instance.load_mm_data = AsyncMock(return_value=SimpleNamespace(videos=[object()]))
    metadata = {'fps': 4, 'total_num_frames': 32, 'duration': 8, 'frames_indices': list(range(0, 32, 2))}
    frames = torch.zeros(16, 3, 32, 32)
    class ReachedProcessor(Exception):
        pass
    instance.process_and_combine_mm_data_async = AsyncMock(side_effect=ReachedProcessor)
    request = SimpleNamespace(video_data=['synthetic'], audio_data=None, rid='video-test')
    with patch.object(qv, 'preprocess_video', AsyncMock(return_value=(frames, metadata))):
        with pytest.raises(ReachedProcessor):
            asyncio.run(instance.process_mm_data_async([], [1, 2], request))
    kwargs = instance.process_and_combine_mm_data_async.call_args.kwargs
    if model_type == 'qwen2_vl':
        assert 'video_metadata' not in kwargs
        assert 'do_sample_frames' not in kwargs
    else:
        assert kwargs['video_metadata'] == [metadata]
        assert kwargs['do_sample_frames'] is False
    assert kwargs['processor_video_config'] == {}


def test_video_timestamp_tokens_positions_and_embedding_slices_match():
    reference = processor('qwen4_exp')
    release = processor('qwen3_8_flash_next')
    prompt = [10, reference.IM_START_TOKEN_ID, reference.VIDEO_TOKEN_ID, reference.IM_END_TOKEN_ID, 11]
    embeddings = {qv.Modality.VIDEO: torch.arange(32, dtype=torch.float32).reshape(16, 2)}
    kwargs = {'video_grid_thw': torch.tensor([[4, 4, 4]]), 'video_timestamps': [[0., 2., 4., 6.]]}
    expected = reference.get_mm_data(prompt, embeddings, **deepcopy(kwargs))
    actual = release.get_mm_data(prompt, embeddings, **deepcopy(kwargs))
    assert actual.input_ids == expected.input_ids
    assert len(actual.mm_items) == len(expected.mm_items) == 4
    assert torch.equal(actual.mrope_positions, expected.mrope_positions)
    assert torch.equal(actual.mrope_position_delta, expected.mrope_position_delta)
    for index, (got, wanted) in enumerate(zip(actual.mm_items, expected.mm_items)):
        assert got.offsets == wanted.offsets
        assert torch.equal(got.precomputed_embeddings, embeddings[qv.Modality.VIDEO][index * 4:(index + 1) * 4])
    assert Tokenizer().encode('<6.0 seconds>')[0] in actual.input_ids


def test_image_offset_positions_match_registered_architecture():
    item = qv.MultimodalDataItem(modality=qv.Modality.IMAGE, offsets=[(2, 5)],
                                model_specific_data={'image_grid_thw': torch.tensor([[1, 4, 4]])})
    kwargs = dict(input_len=8, mm_items=[item], dtype=torch.long, device=torch.device('cpu'))
    expected = processor('qwen4_exp')._compute_image_only_mrope_positions_from_offsets(**kwargs)
    actual = processor('qwen3_8_flash_next')._compute_image_only_mrope_positions_from_offsets(**kwargs)
    assert actual is not None and expected is not None
    assert all(torch.equal(got, wanted) for got, wanted in zip(actual, expected))
