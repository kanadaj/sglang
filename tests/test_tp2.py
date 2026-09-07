"""Bounded real-method TP2 CPU seams; not distributed/GPU certification."""
import ast
import json
import math
import os
from types import SimpleNamespace as NS
import unittest
import torch
from test_packed_ple import runtime, ROOT
from test_integration import functions, source

PREFIX = 'model.layers.0.ple.ple_embedding'


def setup(rank, vocab=255, padded=256, shards=128):
    mod = runtime()
    capacity = padded // 2
    start, end = min(rank * capacity, vocab), min((rank + 1) * capacity, vocab)
    table = mod.PackedPLEStorage(capacity, 32, pin_memory=False)
    emb = NS(tp_size=2, org_vocab_size=vocab, org_vocab_size_padded=padded,
             num_embeddings_padded=padded, num_embeddings_per_partition=capacity,
             num_added_embeddings=0, weight=table.weight, _packed_storage=table,
             _packed_fp8_reference=False,
             shard_indices=NS(org_vocab_start_index=start, org_vocab_end_index=end,
                 padded_org_vocab_start_index=rank*capacity,
                 padded_org_vocab_end_index=(rank+1)*capacity))
    ple = NS(ngram_embedding=emb, _ple_source_packed=True, _ple_global_scale=.3)
    env = dict(torch=torch, os=os, math=math, json=json, ple_modules={PREFIX:ple},
               ple_num_sync_shards=shards, ple_packed_pending={},
               ple_packed_stats={'rows':0,'bytes':0}, loaded_shard_params=set(),
               logger=NS(info=lambda *args: None))
    functions({'load_qwen4_exp_ple_shard','_ple_resolve_global_scale'}, env)
    return table, emb, env


def load_pair(env, idx, vocab=255, shards=128):
    size = (vocab + shards - 1) // shards
    rows = min(size, vocab-idx*size)
    base = f'{PREFIX}.ngram_embedding.shard_{idx}'
    load = env['load_qwen4_exp_ple_shard']
    load(base+'.weight_scale', torch.full((rows,2),1+(idx%4)/2).to(torch.float8_e4m3fn))
    load(base+'.weight', torch.full((rows,16),idx,dtype=torch.uint8))


def finalize(env):
    loop = next(n for n in ast.walk(source()) if isinstance(n,ast.For)
                and isinstance(n.target,ast.Name) and n.target.id=='ple_mod'
                and any(isinstance(x,ast.Attribute) and x.attr=='finalize' for x in ast.walk(n)))
    exec(compile(ast.Module(body=[loop],type_ignores=[]),'<real-finalization>','exec'),env)


class TP2Tests(unittest.TestCase):
    def test_synthetic_metadata_both_ranks_without_payload_allocation(self):
        records = [{'name': f'model.layers.0.ple.ple_embedding.ngram_embedding.shard_{i}.{kind}',
                    'shape': [2, width], 'dtype': dtype}
                   for i in range(128) for kind, width, dtype in [('weight', 80, 'U8'), ('weight_scale', 10, 'F8_E4M3')]]
        records = [r for r in records if '.shard_' in r['name']]
        vocab = sum(r['shape'][0] for r in records if r['name'].endswith('.weight'))
        self.assertEqual(vocab,256)
        actual_prefix = records[0]['name'].split('.ngram_embedding.shard_')[0]
        for rank in range(2):
            table, emb, env = setup(rank)
            capacity = vocab//2
            table.weight = torch.empty((capacity,80),device='meta',dtype=torch.uint8)
            table.scales = torch.empty((capacity,10),device='meta',dtype=torch.float8_e4m3fn)
            emb.org_vocab_size = emb.org_vocab_size_padded = emb.num_embeddings_padded = vocab
            emb.num_embeddings_per_partition = capacity
            emb.shard_indices = NS(org_vocab_start_index=rank*capacity,org_vocab_end_index=(rank+1)*capacity,
                padded_org_vocab_start_index=rank*capacity,padded_org_vocab_end_index=(rank+1)*capacity)
            env['ple_modules'][actual_prefix] = env['ple_modules'].pop(PREFIX)
            for record in sorted(records,key=lambda r:r['name'],reverse=True):
                dtype = torch.uint8 if record['dtype']=='U8' else torch.float8_e4m3fn
                env['load_qwen4_exp_ple_shard'](record['name'],torch.empty(record['shape'],device='meta',dtype=dtype))
            finalize(env)
            self.assertEqual(len(table._ranges),64)
            self.assertEqual(len(table._source_ranges),128)
            self.assertEqual(table.nbytes,capacity*90)

    def test_crossing_shard_simulated_sum_matches_tp1_and_reference(self):
        mod = runtime()
        for vocab,padded,shards in [(255,256,128),(17,24,4)]:
            ranks=[]
            for rank in range(2):
                table, emb, env = setup(rank,vocab,padded,shards)
                for idx in reversed(range(shards)):
                    load_pair(env,idx,vocab,shards)
                finalize(env)
                ranks.append((table,emb))
            weights=torch.cat([table.weight[:emb.shard_indices.org_vocab_end_index-emb.shard_indices.org_vocab_start_index] for table,emb in ranks])
            scales=torch.cat([table.scales.view(torch.uint8)[:emb.shard_indices.org_vocab_end_index-emb.shard_indices.org_vocab_start_index] for table,emb in ranks]).view(torch.float8_e4m3fn)
            boundary=padded//2
            ids=torch.tensor([-2**40,-1,0,boundary-1,boundary,vocab-1,vocab,padded-1,2**40,0])
            codes=torch.stack((weights&15,weights>>4),dim=-1).reshape(vocab,32).long()
            lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
            for reference in [False,True]:
                values=lut[codes]
                scale=scales.float().repeat_interleave(16,dim=1)
                expected_values=((values*(scale*.3)).to(torch.float8_e4m3fn) if reference else values*scale*.3).bfloat16()
                expected=torch.zeros((ids.numel(),32),dtype=torch.bfloat16)
                valid=(ids>=0)&(ids<vocab)
                expected[valid]=expected_values[ids[valid]]
                tp1=torch.empty_like(expected)
                mod.gather_packed_kernel[(ids.numel(),)](weights.data_ptr(),scales.data_ptr(),ids,tp1,.3,32,0,vocab,reference,32,enable_fp_fusion=False)
                outputs=[]
                for table,emb in ranks:
                    out=torch.empty_like(expected)
                    start,end=emb.shard_indices.org_vocab_start_index,emb.shard_indices.org_vocab_end_index
                    mod.gather_packed_kernel[(ids.numel(),)](table.weight.data_ptr(),table.scales.data_ptr(),ids,out,.3,32,start,end,reference,32,enable_fp_fusion=False)
                    owned=(ids>=start)&(ids<end)
                    self.assertTrue(torch.equal(out[owned].view(torch.int16),tp1[owned].view(torch.int16)))
                    self.assertEqual(out[~owned].count_nonzero().item(),0)
                    outputs.append(out)
                torch.testing.assert_close(outputs[0]+outputs[1],tp1,rtol=0,atol=0)
                torch.testing.assert_close(tp1,expected,rtol=0,atol=0)

    def test_offrank_shapes_scales_and_partition_boundaries_fail_closed(self):
        for rank,idx in [(0,127),(1,0)]:
            for rows,cols,kind in [(3,16,'weight'),(0,16,'weight'),(2,8,'weight'),(2,1,'weight_scale')]:
                _,_,env=setup(rank)
                with self.assertRaises(ValueError):
                    env['load_qwen4_exp_ple_shard'](f'{PREFIX}.ngram_embedding.shard_{idx}.{kind}',
                        torch.ones((rows,cols),dtype=torch.uint8 if kind=='weight' else torch.float8_e4m3fn))
                self.assertFalse(env['ple_packed_pending'])
            for scale in [float('nan'),float('inf'),0.,-1.,.5]:
                _,_,env=setup(rank)
                with self.assertRaises(AssertionError):
                    env['load_qwen4_exp_ple_shard'](PREFIX+'.ngram_embedding.weight_scale_2',torch.tensor([scale]))
            for attr,value in [('org_vocab_start_index',1),('org_vocab_end_index',256),('padded_org_vocab_start_index',3),('padded_org_vocab_end_index',129)]:
                _,emb,env=setup(rank)
                setattr(emb.shard_indices,attr,value)
                with self.assertRaisesRegex(ValueError,'layout'):
                    load_pair(env,0)

    def test_missing_half_and_duplicate_pending_offrank_rejected(self):
        _,_,env=setup(0)
        load=env['load_qwen4_exp_ple_shard']
        name=f'{PREFIX}.ngram_embedding.shard_127.weight_scale'
        value=torch.ones((1,2),dtype=torch.float8_e4m3fn)
        load(name,value)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            load(name,value)
        for idx in range(127):
            load_pair(env,idx)
        with self.assertRaisesRegex(ValueError,'global shard count'):
            finalize(env)
        self.assertIn((PREFIX,127),env['ple_packed_pending'])

    def test_malformed_source_identifiers_are_not_silently_ignored(self):
        for suffix in ['-1.weight','01.weight','128.weight','0.weight.extra']:
            _, _, env = setup(1)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                env['load_qwen4_exp_ple_shard'](PREFIX+'.ngram_embedding.shard_'+suffix,
                    torch.ones((2,16),dtype=torch.uint8))

    def test_trailing_newline_shards_rejected_before_local_or_offrank_loading(self):
        for rank in [0, 1]:
            for idx in [0, 127]:
                for kind in ['weight', 'weight_scale']:
                    with self.subTest(rank=rank, idx=idx, kind=kind):
                        table, _, env = setup(rank)
                        rows = 1 if idx == 127 else 2
                        cols = 16 if kind == 'weight' else 2
                        dtype = torch.uint8 if kind == 'weight' else torch.float8_e4m3fn
                        name = f'{PREFIX}.ngram_embedding.shard_{idx}.{kind}\n'
                        with self.assertRaisesRegex(ValueError, 'malformed') as caught:
                            env['load_qwen4_exp_ple_shard'](
                                name, torch.ones((rows, cols), dtype=dtype))
                        self.assertIn(name, str(caught.exception))
                        self.assertFalse(env['ple_packed_pending'])
                        self.assertFalse(env['loaded_shard_params'])
                        self.assertFalse(table._ranges)
                        self.assertFalse(table._source_ranges)
                        # Rejection must not consume the canonical identity.
                        load_pair(env, idx)

    def test_pending_offrank_has_no_payload_and_local_budget_fails_closed(self):
        table, _, env = setup(0)
        load = env['load_qwen4_exp_ple_shard']
        name = f'{PREFIX}.ngram_embedding.shard_127.weight'
        load(name,torch.ones((1,16),dtype=torch.uint8))
        pending = env['ple_packed_pending'][(PREFIX,127)]['weight']
        self.assertEqual(pending.device.type,'meta')
        table.max_pending_bytes = 31
        with self.assertRaisesRegex(ValueError,'pending.*budget'):
            load(f'{PREFIX}.ngram_embedding.shard_0.weight',torch.ones((2,16),dtype=torch.uint8))
        self.assertNotIn((PREFIX,0),env['ple_packed_pending'])

    def test_duplicate_completed_offrank_tensor_rejected_immediately(self):
        for rank, idx in [(0,127),(1,0)]:
            _, _, env = setup(rank)
            load_pair(env,idx)
            rows = 1 if idx==127 else 2
            with self.assertRaisesRegex(ValueError,'duplicate'):
                env['load_qwen4_exp_ple_shard'](f'{PREFIX}.ngram_embedding.shard_{idx}.weight',
                    torch.ones((rows,16),dtype=torch.uint8))

    def test_missing_nonintersecting_pair_fails_global_finalization(self):
        for rank, missing in [(0,127),(1,0)]:
            table, _, env = setup(rank)
            for idx in range(128):
                if idx != missing:
                    load_pair(env,idx)
            self.assertEqual(len(table._ranges),64)
            with self.assertRaisesRegex(ValueError,'global shard count'):
                finalize(env)

    def test_offrank_malformed_dtype_rejected_before_buffering(self):
        for rank, idx in [(0,127),(1,0)]:
            for kind in ['weight','weight_scale']:
                _, _, env = setup(rank)
                rows = 1 if idx==127 else 2
                cols = 16 if kind=='weight' else 2
                with self.assertRaises(ValueError):
                    env['load_qwen4_exp_ple_shard'](f'{PREFIX}.ngram_embedding.shard_{idx}.{kind}',
                        torch.ones((rows,cols),dtype=torch.float32))
                self.assertFalse(env['ple_packed_pending'])

    def test_constructor_allocates_only_local_padded_bytes(self):
        from contextlib import nullcontext
        from typing import Tuple, Optional
        import triton
        from torch import nn
        mod = runtime()
        node = next(n for n in ast.walk(source()) if isinstance(n,ast.ClassDef)
                    and n.name=='Qwen4ExpPinnedHostEmbedding')
        class UnquantizedEmbeddingMethod:
            pass
        allocations = []
        def storage(rows, dim):
            allocations.append((rows,dim))
            return mod.PackedPLEStorage(rows,dim,pin_memory=False)
        scope = dict(torch=torch, nn=nn, triton=triton, os=os, Tuple=Tuple,
                     Optional=Optional, nullcontext=nullcontext,
                     VocabParallelEmbedding=nn.Module,
                     UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,
                     PackedPLEStorage=storage)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-class>','exec'),scope)
        cls = scope['Qwen4ExpPinnedHostEmbedding']
        cls.weight_loader = lambda *args: None
        for rank in range(2):
            _, metadata, _ = setup(rank)
            emb = nn.Module()
            for attr in cls._COPIED_ATTRIBUTES:
                setattr(emb,attr,getattr(metadata,attr,None))
            emb.embedding_dim = 32
            emb.quant_method = UnquantizedEmbeddingMethod()
            emb.weight_scale = torch.ones(1,dtype=torch.bfloat16)
            emb.weight = nn.Parameter(torch.empty((128,32),device='meta',dtype=torch.float8_e4m3fn),requires_grad=False)
            result = cls(emb,packed_nvfp4=True)
            self.assertEqual(result._packed_storage.nbytes,128*18)
            self.assertFalse(hasattr(emb,'weight'))
        self.assertEqual(allocations,[(128,32),(128,32)])

    def test_both_ranks_64_local_128_global_and_padding(self):
        for rank in range(2):
            table, emb, env = setup(rank)
            for idx in reversed(range(128)):
                load_pair(env,idx)
            finalize(env)
            self.assertEqual(len(table._ranges),64)
            self.assertEqual(table.weight.shape[0],128)
            self.assertEqual(table.nbytes,128*18)
            self.assertFalse(env['ple_packed_pending'])
            start = emb.shard_indices.org_vocab_start_index
            for local in range(emb.shard_indices.org_vocab_end_index-start):
                self.assertEqual(table.weight[local,0].item(),(start+local)//2)
