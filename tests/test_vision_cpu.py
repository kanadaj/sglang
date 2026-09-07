"""Execute the real vision adapter classes on CPU with explicit parent seams.

Tests layout/bias ownership, not CUTLASS/Marlin numerics or CUDA execution.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]

def adapters():
    class FP8Parent:
        use_mxfp8 = True
        def process_weights_after_loading(self, layer):
            layer.parent_shape = tuple(layer.weight.shape)
        def apply(self, layer, x, bias=None):
            assert bias is None
            return torch.nn.functional.linear(x, layer.weight)
    class MarlinParent:
        def process_weights_after_loading(self, layer):
            if layer.bias is not None:
                layer.bias = torch.nn.Parameter(layer.bias.flip(0), requires_grad=False)
    source = ROOT / 'runtime/python/sglang/srt/layers/quantization/vision_mxfp8.py'
    tree = ast.parse(source.read_text())
    classes: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace: dict[str, Any] = dict(torch=torch, logger=logging.getLogger(__name__),
                     Fp8LinearMethod=FP8Parent, ModelOptNvFp4A16LinearMethod=MarlinParent)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace

class VisionTests(unittest.TestCase):
    def test_fc1_padding_preserves_weights_scales_and_slices_before_bias(self):
        method = adapters()['VisionMxfp8PaddedLinearMethod']()
        w = torch.arange(48 * 32, dtype=torch.float32).reshape(48, 32) / 1000
        scales = torch.full((48, 1), 129, dtype=torch.uint8)
        layer = NS(weight=torch.nn.Parameter(w.clone()), weight_scale_inv=NS(data=scales.clone()))
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.parent_shape, (64, 32))
        torch.testing.assert_close(layer.weight[:48], w)
        self.assertEqual(torch.count_nonzero(layer.weight[48:]).item(), 0)
        torch.testing.assert_close(layer.weight_scale_inv.data[:48], scales)
        self.assertTrue(torch.all(layer.weight_scale_inv.data[48:] == 127))
        x = torch.ones(2, 32)
        bias = torch.arange(48, dtype=torch.float32)
        torch.testing.assert_close(method.apply(layer, x, bias), torch.nn.functional.linear(x, w, bias))
    def test_fc1_rejects_wrong_group_layout(self):
        method = adapters()['VisionMxfp8PaddedLinearMethod']()
        layer = NS(weight=NS(data=torch.ones(48, 31)), weight_scale_inv=NS(data=torch.ones(48, 1)))
        with self.assertRaises(ValueError):
            method.process_weights_after_loading(layer)
    def test_fc2_restores_logical_order_and_handles_no_bias(self):
        for bias in [None, torch.nn.Parameter(torch.arange(128, dtype=torch.float32))]:
            layer = NS(output_size_per_partition=128, bias=bias)
            adapters()['VisionNvFp4A16LinearMethod']().process_weights_after_loading(layer)
            if bias is None:
                self.assertIsNone(layer.bias)
            else:
                torch.testing.assert_close(layer.bias, bias)
                self.assertFalse(layer.bias.requires_grad)
    def test_fc2_rejects_unproven_output_padding(self):
        with self.assertRaises(ValueError):
            adapters()['VisionNvFp4A16LinearMethod']().process_weights_after_loading(NS(output_size_per_partition=129))
