"""Actual reduction/prefetch methods with CPU stream/collective substitutes.

These test call ownership and ordering, NOT NCCL or CUDA stream correctness.
"""
import ast
from contextlib import nullcontext
from types import SimpleNamespace as NS
from typing import Optional
import unittest
import torch
from test_integration import source


class CollectiveSeamTests(unittest.TestCase):
    def test_normal_and_prefetch_reduce_exactly_once_or_reject_scattered(self):
        tree=source()
        pinned=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpPinnedHostEmbedding')
        nodes=[n for n in pinned.body if isinstance(n,ast.FunctionDef) and n.name in {'reduce','forward'}]
        nodes += [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in {'start_prefetch','_consume_prefetched_embeddings','_finish_embedding_lookup'}]
        for tp_size, packed, scattered, attention_group in __import__('itertools').product([1,2], [False,True], [False,True], [False,True]):
            with self.subTest(tp_size=tp_size, packed=packed, scattered=scattered, attention_group=attention_group):
                calls=[]
                def collective(name):
                    def run(value):
                        calls.append(name)
                        return value*2
                    return run
                stream=NS(wait_stream=lambda other: calls.append('wait'))
                fake_torch=NS(Tensor=torch.Tensor,cuda=NS(current_stream=lambda:stream,stream=lambda s:nullcontext()))
                env=dict(torch=fake_torch,Optional=Optional,_PLEBatch=NS,ForwardBatch=NS,
                         get_attn_tp_context=lambda:NS(input_scattered=scattered),
                         attn_tp_all_reduce=collective('attn_reduce'),
                         tensor_model_parallel_all_reduce=collective('tp_reduce'))
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-collective-methods>','exec'),env)
                ids=NS(shape=(2,1),record_stream=lambda s:calls.append('record'))
                def gather(ids,out=None):
                    calls.append('gather')
                    if out is None:
                        return torch.ones((2,1,32),dtype=torch.bfloat16)
                    out.fill_(1)
                    return out
                emb=NS(tp_size=tp_size,_packed_storage=NS() if packed else None,
                       use_attn_tp_group=attention_group,gather=gather,weight_scale=torch.ones(1))
                emb.reduce=lambda value:env['reduce'](emb,value)
                def compute(batch):
                    calls.append('compute')
                    return ids
                def prepare(*args):
                    calls.append('prepare')
                    return ids,2
                def buffer(*args):
                    calls.append('buffer')
                    return torch.empty((2,32),dtype=torch.bfloat16)
                ple=NS(ngram_embedding=emb,ngram_heads=1,gather_dp_tokens=False,
                       compute_ngram_ids=compute,_prepare_embedding_lookup=prepare)
                ple._finish_embedding_lookup=lambda *args:env['_finish_embedding_lookup'](ple,*args)
                layer=NS(_prefetch_stream=stream,_prefetch_state=None,ple_embedding=ple,
                         _get_prefetch_buffer=buffer)
                if tp_size == 2 and packed and scattered:
                    for path in ['forward','start_prefetch']:
                        with self.subTest(path=path):
                            with self.assertRaisesRegex(RuntimeError,'packed TP2 PLE.*input_scattered'):
                                if path == 'forward':
                                    env[path](emb,ids)
                                else:
                                    env[path](layer,NS(physical_tokens=2),NS())
                            self.assertEqual(calls,[])
                            self.assertIsNone(layer._prefetch_state)
                    continue
                direct=env['forward'](emb,ids)
                expected_reduces=(['attn_reduce' if attention_group else 'tp_reduce']
                                  if tp_size > 1 and not scattered else [])
                self.assertEqual([c for c in calls if c.endswith('reduce')],expected_reduces)
                self.assertEqual(calls.count('gather'),1)
                self.assertEqual(calls,['gather']+expected_reduces)
                calls.clear()
                env['start_prefetch'](layer,NS(physical_tokens=2),NS())
                self.assertEqual(calls.count('gather'),1)
                self.assertFalse([c for c in calls if c.endswith('reduce')])
                prefetched=env['_consume_prefetched_embeddings'](layer,NS())
                self.assertEqual([c for c in calls if c.endswith('reduce')],expected_reduces)
                if expected_reduces:
                    self.assertLess(calls.index('gather'),calls.index(expected_reduces[0]))
                    self.assertEqual(calls[calls.index(expected_reduces[0])-1],'wait')
                self.assertIsNone(layer._prefetch_state)
                torch.testing.assert_close(prefetched,direct.reshape(2,32).float(),rtol=0,atol=0)
                with self.assertRaisesRegex(RuntimeError,'missing'):
                    env['_consume_prefetched_embeddings'](layer,NS())
                self.assertEqual([c for c in calls if c.endswith('reduce')],expected_reduces)
