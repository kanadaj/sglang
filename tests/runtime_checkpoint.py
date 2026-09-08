"""GPU-free full-module loader/packed-PLE regression. Run inside the runtime image.
Only model construction, post-load GPU offload and runtime metadata are substituted.
HF resolution, index filtering, lazy safetensors iteration and PLE loading are real.
"""
import json
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
from sglang.srt.model_loader import loader as lm
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as Qwen, Qwen4ExpNGramEmbedding as NGram
from sglang.srt.models.packed_ple import PackedPLEStorage

REPO = 'fixture/packed-ple'
PREFIX = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
KEY = PREFIX + '.weight_scale_2'

def fixture(folder, scale=.3, missing=False):
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for i in range(2):
        tensors[f'{PREFIX}.shard_{i}.weight'] = torch.full((2,16), 0x32, dtype=torch.uint8)
        tensors[f'{PREFIX}.shard_{i}.weight_scale'] = torch.ones((2,2), dtype=torch.float8_e4m3fn)
    save_file(tensors, str(folder/'a.safetensors'))
    scales = {} if missing else {KEY: torch.as_tensor(scale).reshape(-1)}
    save_file(scales, str(folder/'z.safetensors'))
    wm = {k:'a.safetensors' for k in tensors}
    wm.update({k:'z.safetensors' for k in scales})
    (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map':wm}))
    return folder

def model(config_path=REPO):
    m = Qwen.__new__(Qwen)
    nn.Module.__init__(m)
    m.config = NS(_name_or_path=config_path, split_ngram_parts=2)
    m.language_model_only = False
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    m.model.layers[1].ple = nn.Module()
    ple = NGram.__new__(NGram)
    nn.Module.__init__(ple)
    m.model.layers[1].ple.ple_embedding = ple
    emb = nn.Module()
    ple.ngram_embedding = emb
    ple._ple_source_packed = True
    emb._packed_storage = PackedPLEStorage(4,32,pin_memory=False)
    emb.weight = nn.Parameter(emb._packed_storage.weight, requires_grad=False)
    emb.org_vocab_size = 4
    emb.embedding_dim = 32
    emb._packed_fp8_reference = True
    emb.shard_indices = NS(org_vocab_start_index=0, org_vocab_end_index=4)
    m.post_load_weights = lambda: None  # GPU host-offload conversion is not under test.
    m._log_weight_dtype_census = lambda: None
    return m

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)/'nondefault-cache'
        self.a = fixture(self.cache/'models--fixture--packed-ple/snapshots'/('a'*40))
        self.b = fixture(self.cache/'models--fixture--packed-ple/snapshots'/('b'*40), .7)
        self.loader = lm.DefaultModelLoader(LoadConfig(load_format=LoadFormat.SAFETENSORS,
            download_dir=str(self.cache), model_loader_extra_config={'enable_multithread_load':False}))
        self.meta = NS(weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False,
            weight_loader_prefetch_num_threads=1, weight_loader_drop_cache_after_load=False)
        for p in [patch.object(lm, 'get_server_args', return_value=None),
                  patch.object(lm, 'get_model', return_value=self.meta),
                  patch('sglang.srt.server_args.get_global_server_args', return_value=NS(model_path=REPO)),
                  patch.object(lm, '_get_quantization_config', return_value=None)]:
            p.start(); self.addCleanup(p.stop)
        os.environ['SGLANG_SORT_WEIGHT_FILES'] = '0'

    def config(self, path=REPO, revision='a'*40):
        return NS(model_path=path, revision=revision, dtype=torch.bfloat16,
                  hf_config=NS(architectures=['Qwen4ExpForConditionalGeneration']))

    def load(self, m, cfg):
        with patch.object(lm, '_initialize_model', return_value=m):
            return self.loader.load_model(model_config=cfg, device_config=NS(device='cpu'))

    def check_scale(self, m, scale):
        storage = m.model.layers[1].ple.ple_embedding.ngram_embedding._packed_storage
        self.assertEqual(storage.global_scale, float(torch.tensor(scale)))
        self.assertEqual(storage.weight.shape, (4,16))
        self.assertTrue(torch.all(storage.weight == 0x32))

    def test_hf_id_real_offline_resolution_late_nonunit_scale(self):
        m = model()
        self.load(m, self.config())
        self.check_scale(m, .3)
        self.assertEqual(m.config._name_or_path, REPO)

    def test_load_metadata_is_consumed_even_on_scale_failure(self):
        for missing in [False, True]:
            fixture(self.a, missing=missing)
            m = model()
            try:
                self.load(m, self.config())
            except ValueError:
                if not missing:
                    raise
            self.assertFalse(hasattr(m, '_resolved_weight_sources'))

    def test_empty_resolved_files_fail_closed(self):
        m = model()
        (self.a/'model.safetensors.index.json').unlink()
        cfg = self.config(str(self.a))
        m._resolved_weight_sources = (lm.DefaultModelLoader.ResolvedSource(
            lm.DefaultModelLoader.Source.init_new(cfg, m), str(self.a), (), True),)
        from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
        with self.assertRaisesRegex(ValueError, 'not found in the checkpoint'):
            m.load_weights(safetensors_weights_iterator([str(self.a/'a.safetensors')]))

    def test_local_checkpoint(self):
        m = model(str(self.a))
        self.load(m, self.config(str(self.a)))
        self.check_scale(m, .3)

    def test_direct_local_load_without_loader_metadata(self):
        from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
        m = model(str(self.a))
        m.load_weights(safetensors_weights_iterator([str(self.a/'a.safetensors'),str(self.a/'z.safetensors')]))
        self.check_scale(m, .3)

    def test_target_draft_revision_identity_survives_interleaving(self):
        target, draft = model(), model(str(self.a))  # stale local config must not win
        tw = self.loader._get_all_weights(self.config(), target)
        dw = self.loader._get_all_weights(self.config(revision='b'*40), draft)
        self.assertNotEqual(target._resolved_weight_sources[0].hf_folder,
                            draft._resolved_weight_sources[0].hf_folder)
        draft.load_weights(dw)
        target.load_weights(tw)
        self.check_scale(target, .3)
        self.check_scale(draft, .7)

    def test_resolves_before_load_weights_but_tensor_iteration_stays_lazy(self):
        m = model()
        with patch.object(lm, 'safetensors_weights_iterator', wraps=lm.safetensors_weights_iterator) as read:
            with patch.object(self.loader, '_prepare_weights', wraps=self.loader._prepare_weights) as prepare:
                weights = self.loader._get_all_weights(self.config(), m)
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(read.call_count, 0)
                self.assertEqual(m._resolved_weight_sources[0].hf_folder, str(self.a))
                m.load_weights(weights)
                self.assertEqual(prepare.call_count, 1)
        self.check_scale(m, .3)

    def test_secondary_source_keeps_prefix_and_primary_ple_identity(self):
        m = model()
        m.secondary_weights = (lm.DefaultModelLoader.Source(REPO, 'b'*40, prefix='mtp.secondary.'),)
        with patch.object(self.loader, '_prepare_weights', wraps=self.loader._prepare_weights) as prepare:
            weights = self.loader._get_all_weights(self.config(), m)
            sources = m._resolved_weight_sources
            self.assertEqual(len(sources), 2)
            self.assertEqual(sources[0].hf_folder, str(self.a))
            self.assertEqual(sources[1].hf_folder, str(self.b))
            secondary = list(self.loader._get_weights_iterator(sources[1].source, resolved_source=sources[1]))
            self.assertTrue(secondary)
            self.assertTrue(all(n.startswith('mtp.secondary.') for n, _ in secondary))
            m.load_weights(weights)  # Qwen deliberately ignores MTP auxiliary weights.
            self.assertEqual(prepare.call_count, 2)
        self.check_scale(m, .3)

    def test_pre_resolved_startup_commit_does_not_resolve_again(self):
        m = model(str(self.a))
        cfg = self.config(revision='b'*40)
        sources = self.loader.resolve_model_weights(cfg, m)
        with patch.object(self.loader, '_prepare_weights', side_effect=AssertionError('second resolution')):
            self.loader.commit_model_weights(model=m, model_config=cfg, resolved_sources=sources,
                target_device=torch.device('cpu'), startup_prefetch_active=False)
        self.check_scale(m, .7)

    def test_local_subdirectory_is_used_verbatim(self):
        folder = fixture(Path(self.tmp.name)/'download-dir/revision/subfolder', .6)
        m = model()
        self.load(m, self.config(str(folder)))
        self.check_scale(m, .6)

    def test_missing_index_key_fails_closed(self):
        fixture(self.a, missing=True)
        with self.assertRaisesRegex(ValueError, 'missing from the checkpoint index'):
            self.load(model(), self.config())

    def test_index_key_missing_from_tensor_file_fails_closed(self):
        save_file({}, str(self.a/'z.safetensors'))
        with self.assertRaisesRegex(ValueError, 'not found in the checkpoint'):
            self.load(model(), self.config())

    def test_invalid_global_scales_fail_closed(self):
        for scale in [0., -1., float('nan'), float('inf'), [1.,2.]]:
            with self.subTest(scale=scale):
                fixture(self.a, scale)
                with self.assertRaisesRegex(AssertionError, 'scalar|finite and positive'):
                    self.load(model(), self.config())

    def test_changed_index_cannot_read_unselected_revision_file(self):
        m = model()
        weights = self.loader._get_all_weights(self.config(), m)
        index = self.a/'model.safetensors.index.json'
        data = json.loads(index.read_text())
        data['weight_map'][KEY] = str(self.b/'z.safetensors')
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'not in the loader-resolved checkpoint'):
            m.load_weights(weights)

    def test_no_index_uses_actual_selected_safetensors_names(self):
        (self.a/'model.safetensors.index.json').unlink()
        m = model()
        self.load(m, self.config(str(self.a)))
        self.check_scale(m, .3)

    def test_no_resolved_path_never_falls_back_to_target_server(self):
        from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
        with patch('sglang.srt.server_args.get_global_server_args', return_value=NS(model_path=str(self.b))):
            with self.assertRaisesRegex(ValueError, 'no local checkpoint dir'):
                model().load_weights(safetensors_weights_iterator([str(self.a/'a.safetensors')]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
