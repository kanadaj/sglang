"""Exercise real patched SGLang methods, isolated from CUDA/TP boot imports."""
import ast
from contextlib import nullcontext
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from typing import Tuple, Optional
import torch
from torch import nn
import triton
from safetensors.torch import save_file
from test_packed_ple import runtime, ROOT


def source():
    p = ROOT / 'runtime/python/sglang/srt/models/qwen4_exp.py'
    if not p.exists():
        p = ROOT / 'reference/sg_qwen4_exp.py'
    return ast.parse(p.read_text())


def functions(names, scope):
    scope.setdefault("resolved_sources", ())  # load_weights closure, direct local fixtures
    nodes = [n for n in ast.walk(source()) if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), '<actual-sglang-methods>', 'exec'),scope)


class IntegrationTests(unittest.TestCase):
    def test_opt_in_constructs_only_meta_table_and_keeps_default_unchanged(self):
        init=next(n for n in ast.walk(source()) if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpNGramEmbedding').body
        init=next(n for n in init if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        start=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ple_embedding_dtype' for t in n.targets))
        code=compile(ast.fix_missing_locations(ast.Module(body=init.body[start:],type_ignores=[])),'<actual-ngram-constructor-tail>','exec')
        def vocab(rows,dim,**kwargs):
            emb=nn.Module()
            emb.weight=nn.Parameter(torch.empty((rows,dim),dtype=kwargs['params_dtype']),requires_grad=False)
            return emb
        for enabled, offload, expected in [('1',True,'meta'),('0',True,'cpu'),('1',False,'cpu')]:
            obj=SimpleNamespace(head_dim_per_ngram=32,use_attn_tp_ngram=False)
            env=dict(self=obj,config=SimpleNamespace(ple_embedding_dtype='nvfp4',ple_offload_embedding=offload),
                     torch=torch,os=os,nullcontext=nullcontext,quant_config=None,padded_vocab_size=4,VocabParallelEmbedding=vocab)
            with patch.dict(os.environ,{'SGLANG_PLE_PACKED_NVFP4':enabled}): exec(code,env)
            self.assertEqual(obj.ngram_embedding.weight.device.type,expected)
            self.assertEqual(obj.ngram_embedding.weight_scale.device.type,'cpu')

    def test_pinned_class_constructs_packed_and_gathers_to_prefetch_output(self):
        mod = runtime()
        node = next(n for n in ast.walk(source()) if isinstance(n,ast.ClassDef) and n.name == 'Qwen4ExpPinnedHostEmbedding')
        class UnquantizedEmbeddingMethod: pass
        env = dict(torch=torch,nn=nn,triton=triton,os=os,Tuple=Tuple,Optional=Optional,
                   nullcontext=nullcontext,VocabParallelEmbedding=nn.Module,
                   UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,
                   PackedPLEStorage=lambda rows, dim: mod.PackedPLEStorage(rows,dim,pin_memory=False),
                   gather_packed_kernel=mod.gather_packed_kernel)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<actual-pinned-class>', 'exec'),env)
        cls=env['Qwen4ExpPinnedHostEmbedding']
        self.assertIn('packed_nvfp4',inspect.signature(cls).parameters, 'packed constructor is not integrated')
        # Only distributed metadata and pinning are substituted; real storage,
        # constructor, gather, output checks, and Triton kernel are exercised.
        cls.weight_loader=lambda *args: None
        emb=nn.Module()
        for attr in cls._COPIED_ATTRIBUTES:
            setattr(emb,attr,None)
        emb.tp_size=1
        emb.num_added_embeddings=0
        emb.embedding_dim=32
        emb.shard_indices=SimpleNamespace(org_vocab_start_index=10,org_vocab_end_index=13)
        emb.weight=nn.Parameter(torch.empty((3,32),device='meta',dtype=torch.float8_e4m3fn),requires_grad=False)
        emb.quant_method=UnquantizedEmbeddingMethod()
        emb.weight_scale=torch.ones(1,dtype=torch.bfloat16)
        pinned=cls(emb,packed_nvfp4=True)
        self.assertEqual(tuple(pinned.weight.shape),(3,16))
        self.assertEqual(pinned.weight.device.type,'cpu')
        weights=torch.full((3,16),0x32,dtype=torch.uint8)
        scales=torch.ones((3,2),dtype=torch.float8_e4m3fn)
        pinned._packed_storage.load(weights,scales,10,10,13,.3)
        ids=torch.tensor([9,10,12,13])
        output=torch.empty((4,32),dtype=torch.bfloat16)
        self.assertIs(pinned.gather(ids,out=output),output)
        expected=torch.tensor([1.,1.5]*16).mul(.3).bfloat16()
        torch.testing.assert_close(output[1],expected,rtol=0,atol=0)
        torch.testing.assert_close(output[2],expected,rtol=0,atol=0)
        self.assertEqual(output[[0,3]].count_nonzero().item(),0)
        strided_ids = torch.tensor([9,99,10,99,12,99,13,99])[::2]
        torch.testing.assert_close(pinned.gather(strided_ids),output,rtol=0,atol=0)
        self.assertEqual(tuple(pinned.gather(torch.empty(0,dtype=torch.int64)).shape),(0,32))
        with self.assertRaises(ValueError): pinned.gather(ids,out=output.float())

    def test_gather_normalizes_strided_ids_before_either_kernel(self):
        env = dict(torch=torch, Optional=Optional)
        functions({'gather'}, env)
        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    ids = args[2] if len(args) > 4 else args[1]
                    self.assert_ids(ids)
                return launch
        kernel = Kernel()
        kernel.assert_ids = lambda ids: self.assertTrue(ids.is_contiguous())
        env.update(gather_packed_kernel=kernel, _gather_ple_embedding_from_pinned_kernel=kernel)
        ids = torch.tensor([10,99,11,99,12,99])[::2]
        for packed in [True, False]:
            obj = SimpleNamespace(embedding_dim=32, _block_d=32, _packed_fp8_reference=False,
                weight=torch.empty((3,32)), shard_indices=SimpleNamespace(org_vocab_start_index=10,org_vocab_end_index=13),
                _packed_storage=SimpleNamespace(weight=torch.empty(1),scales=torch.empty(1),global_scale=1.) if packed else None,
                allocate_output=lambda shape, device: torch.empty(shape,dtype=torch.bfloat16,device=device))
            with self.subTest(packed=packed):
                env['gather'](obj, ids)

    def test_gather_rejects_noncontiguous_output_before_launch(self):
        env = dict(torch=torch, Optional=Optional)
        functions({'gather'}, env)
        class NoLaunch:
            def __getitem__(self, grid):
                raise AssertionError('unsafe output reached kernel launch')
        env.update(gather_packed_kernel=NoLaunch(), _gather_ple_embedding_from_pinned_kernel=NoLaunch())
        ids = torch.tensor([0,1,2])
        for packed in [True, False]:
            obj = SimpleNamespace(embedding_dim=32, _packed_storage=SimpleNamespace(global_scale=1.) if packed else None)
            outputs = [torch.empty((1,32),dtype=torch.bfloat16).expand(3,32),
                       torch.empty((32,3),dtype=torch.bfloat16).t(),
                       torch.empty((3,64),dtype=torch.bfloat16)[:,::2]]
            for output in outputs:
                with self.subTest(packed=packed,stride=output.stride()), self.assertRaisesRegex(ValueError, 'contiguous'):
                    env['gather'](obj, ids, out=output)

    def test_packed_loader_rejects_malformed_shards_before_buffering(self):
        mod = runtime()
        prefix = 'model.layers.0.ple.ple_embedding'
        # 128 shards, with a short final shard (255 total rows).
        cases = [(128,2,16,'weight'), (0,3,16,'weight'), (0,1,16,'weight'),
                 (127,2,16,'weight'), (0,2,8,'weight'), (0,2,1,'weight_scale')]
        for idx, rows, cols, kind in cases:
            table = mod.PackedPLEStorage(255,32,pin_memory=False)
            emb = SimpleNamespace(weight=table.weight,_packed_storage=table,org_vocab_size=255,embedding_dim=32,
                shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=255))
            ple = SimpleNamespace(ngram_embedding=emb,_ple_source_packed=True,_ple_global_scale=1.)
            env = dict(torch=torch,os=os,math=math,json=json,ple_modules={prefix:ple},ple_num_sync_shards=128,
                ple_packed_pending={},ple_packed_stats={'rows':0,'bytes':0},loaded_shard_params=set())
            functions({'load_qwen4_exp_ple_shard','_ple_resolve_global_scale','copy_ple_rows_to_tp_embedding'},env)
            value = torch.ones((rows,cols),dtype=torch.uint8 if kind=='weight' else torch.float8_e4m3fn)
            with self.subTest(idx=idx,shape=value.shape,kind=kind):
                with self.assertRaises(ValueError):
                    env['load_qwen4_exp_ple_shard'](f'{prefix}.ngram_embedding.shard_{idx}.{kind}',value)
                self.assertFalse(env['ple_packed_pending'])

    def test_packed_loader_requires_complete_tp1_global_layout(self):
        mod = runtime()
        prefix = 'model.layers.0.ple.ple_embedding'
        table = mod.PackedPLEStorage(254,32,pin_memory=False)
        emb = SimpleNamespace(weight=table.weight,_packed_storage=table,org_vocab_size=256,
            shard_indices=SimpleNamespace(org_vocab_start_index=1,org_vocab_end_index=255))
        env = dict(torch=torch,ple_modules={prefix:SimpleNamespace(ngram_embedding=emb,_ple_source_packed=True)},
            ple_num_sync_shards=128,ple_packed_pending={})
        functions({'load_qwen4_exp_ple_shard'},env)
        with self.assertRaisesRegex(ValueError, 'TP1'):
            env['load_qwen4_exp_ple_shard'](prefix+'.ngram_embedding.shard_0.weight',torch.ones((2,16),dtype=torch.uint8))

    def test_packed_loader_requires_exactly_128_complete_shard_pairs(self):
        mod = runtime()
        prefix = 'model.layers.0.ple.ple_embedding'
        # Permit allocation padding, but require all 255 original vocabulary rows.
        table = mod.PackedPLEStorage(256,32,pin_memory=False)
        emb = SimpleNamespace(weight=table.weight,_packed_storage=table,org_vocab_size=255,
            shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=255))
        env = dict(torch=torch,os=os,math=math,json=json,
            ple_modules={prefix:SimpleNamespace(ngram_embedding=emb,_ple_source_packed=True,_ple_global_scale=1.)},
            ple_num_sync_shards=128,ple_packed_pending={},ple_packed_stats={'rows':0,'bytes':0},loaded_shard_params=set())
        functions({'load_qwen4_exp_ple_shard','_ple_resolve_global_scale'},env)
        load = env['load_qwen4_exp_ple_shard']
        for idx in reversed(range(128)):
            rows = 1 if idx == 127 else 2
            name = f'{prefix}.ngram_embedding.shard_{idx}'
            scales = torch.ones((rows,2),dtype=torch.float8_e4m3fn)
            load(name+'.weight_scale',scales)
            if idx == 127:
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    load(name+'.weight_scale',scales)
            load(name+'.weight',torch.full((rows,16),idx,dtype=torch.uint8))
            if idx == 1:
                with self.assertRaises(ValueError): table.finalize(0,255)
        table.finalize(0,255,expected_shards=128)
        self.assertEqual(len(table._ranges),128)
        self.assertEqual(env['ple_packed_stats']['rows'],255)
        self.assertFalse(env['ple_packed_pending'])
        for idx in range(128):
            self.assertEqual(table.weight[idx*2,0].item(),idx)
        with self.assertRaises(ValueError):
            load(prefix+'.ngram_embedding.shard_0.weight_scale',torch.ones((2,2),dtype=torch.float8_e4m3fn))
            load(prefix+'.ngram_embedding.shard_0.weight',torch.ones((2,16),dtype=torch.uint8))
            table.finalize(0,255)

    def test_synthetic_128_shard_shapes_pass_source_validation(self):
        import re
        records = [{'name': f'model.layers.0.ple.ple_embedding.ngram_embedding.shard_{i}.{kind}',
                    'shape': [2 if i < 127 else 1, width], 'dtype': dtype}
                   for i in range(128) for kind, width, dtype in [('weight', 80, 'U8'), ('weight_scale', 10, 'F8_E4M3')]]
        shards = [r for r in records if '.shard_' in r['name']]
        weights = [r for r in shards if r['name'].endswith('.weight')]
        self.assertEqual(len(weights),128)
        self.assertEqual(len(shards),256)
        rows = sum(r['shape'][0] for r in weights)
        prefix = weights[0]['name'].split('.ngram_embedding.shard_')[0]
        storage = SimpleNamespace(weight=torch.empty((rows,80),device='meta',dtype=torch.uint8),
                                  scales=torch.empty((rows,10),device='meta',dtype=torch.float8_e4m3fn))
        emb = SimpleNamespace(_packed_storage=storage,org_vocab_size=rows,
            shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=rows))
        env = dict(torch=torch,ple_modules={prefix:SimpleNamespace(ngram_embedding=emb,_ple_source_packed=True)},
                   ple_num_sync_shards=128,ple_packed_pending={})
        functions({'load_qwen4_exp_ple_shard'},env)
        self.assertEqual({int(re.search(r'shard_(\d+)',r['name'])[1]) for r in weights},set(range(128)))
        for record in shards:
            env['ple_packed_pending'].clear()  # Validate synthetic headers without allocating checkpoint data.
            dtype = torch.uint8 if record['dtype']=='U8' else torch.float8_e4m3fn
            self.assertTrue(env['load_qwen4_exp_ple_shard'](record['name'],torch.empty(record['shape'],device='meta',dtype=dtype)))

    def test_loader_finalization_rejects_wrong_configured_shard_count(self):
        mod = runtime()
        table = mod.PackedPLEStorage(2,32,pin_memory=False)
        table.load(torch.ones((2,16),dtype=torch.uint8),torch.ones((2,2),dtype=torch.float8_e4m3fn),0,0,2,1.)
        loop = next(n for n in ast.walk(source()) if isinstance(n,ast.For)
                    and isinstance(n.target,ast.Name) and n.target.id=='ple_mod'
                    and any(isinstance(x,ast.Attribute) and x.attr=='finalize' for x in ast.walk(n)))
        emb = SimpleNamespace(_packed_storage=table,_packed_fp8_reference=False,
            shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=2))
        env = dict(ple_modules={'ple':SimpleNamespace(ngram_embedding=emb)},ple_num_sync_shards=128,
                   logger=SimpleNamespace(info=lambda *args:None))
        with self.assertRaisesRegex(ValueError, 'shard count'):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'<actual-finalization>','exec'),env)

    def test_actual_shard_loader_reversed_pairs_late_global_scale(self):
        mod=runtime()
        table=mod.PackedPLEStorage(4,32,pin_memory=False)
        prefix='model.layers.0.ple.ple_embedding'
        emb=SimpleNamespace(weight=table.weight,_packed_storage=table,org_vocab_size=4,
                            shard_indices=SimpleNamespace(org_vocab_start_index=0,org_vocab_end_index=4))
        ple=SimpleNamespace(ngram_embedding=emb,_ple_source_packed=True)
        with tempfile.TemporaryDirectory() as d:
            key=prefix.replace('model.layers.','model.language_model.layers.',1)+'.ngram_embedding.weight_scale_2'
            save_file({key:torch.tensor([.3])},str(Path(d)/'global.safetensors'))
            (Path(d)/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'global.safetensors'}}))
            env=dict(torch=torch,os=os,math=math,json=json,ple_modules={prefix:ple},
                     self=SimpleNamespace(config=SimpleNamespace(_name_or_path=d)),
                     ple_packed_pending={},ple_packed_stats={'rows':0,'bytes':0},
                     ple_num_sync_shards=2,loaded_shard_params=set(),
                     _PLE_E2M1_LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]))
            functions({'load_qwen4_exp_ple_shard','_ple_resolve_global_scale','copy_ple_rows_to_tp_embedding'},env)
            load=env['load_qwen4_exp_ple_shard']
            for shard in [1,0]:
                base=f'{prefix}.ngram_embedding.shard_{shard}'
                weights=torch.full((2,16),0x32+shard,dtype=torch.uint8)
                scales=torch.full((2,2),1.+shard,dtype=torch.float32).to(torch.float8_e4m3fn)
                self.assertTrue(load(base+'.weight_scale',scales))
                self.assertTrue(load(base+'.weight',weights))
                torch.testing.assert_close(table.weight[shard*2:shard*2+2],weights)
            self.assertTrue(load(prefix+'.ngram_embedding.weight_scale_2',torch.tensor([.3])))
            self.assertEqual(table.global_scale,float(torch.tensor(.3)))
            table.finalize(0,4)
            self.assertFalse(env['ple_packed_pending'])
            # Disable the new storage: the exact old expanded-FP8 loader still runs.
            emb._packed_storage=None
            emb.weight=torch.empty((4,32),dtype=torch.float8_e4m3fn)
            for shard in [0,1]:
                base=f'{prefix}.ngram_embedding.shard_{shard}'
                weights=torch.full((2,16),0x32,dtype=torch.uint8)
                scales=torch.ones((2,2),dtype=torch.float8_e4m3fn)
                self.assertTrue(load(base+'.weight',weights))
                self.assertTrue(load(base+'.weight_scale',scales))
            expected=torch.tensor([1.,1.5]*16).mul(float(torch.tensor(.3))).to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(emb.weight.float(),expected.expand(4,32),rtol=0,atol=0)
