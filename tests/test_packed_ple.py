import importlib.util
import pathlib
import unittest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def runtime():
    path = ROOT / 'runtime/python/sglang/srt/models/packed_ple.py'
    assert path.exists(), 'packed PLE runtime is not implemented'
    spec = importlib.util.spec_from_file_location('packed_ple', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackedPLETests(unittest.TestCase):
    def test_kernel_all_nibbles_group_scales_global_and_row_selection(self):
        mod = runtime()
        packed = torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]*2]*3, dtype=torch.uint8)
        scales = torch.tensor([[0.5,2.0],[1.5,0.25],[4.0,0.125]], dtype=torch.float8_e4m3fn)
        ids = torch.tensor([[12,10,11,12]], dtype=torch.int64)
        out = torch.empty((*ids.shape,32), dtype=torch.bfloat16)
        mod.gather_packed_kernel[(ids.numel(),)](packed.data_ptr(), scales.data_ptr(), ids, out, 0.3, 32, 10, 13, False, 32)
        lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
        expected = torch.stack([(lut.repeat(2)*scales.float()[i].repeat_interleave(16)*0.3).bfloat16() for i in [2,0,1,2]]).reshape_as(out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_storage_retains_bytes_copies_overlap_and_preserves_global_scale(self):
        mod = runtime()
        self.assertTrue(hasattr(mod, 'PackedPLEStorage'), 'packed storage is not implemented')
        table = mod.PackedPLEStorage(3, 32, pin_memory=False)
        packed = torch.arange(80, dtype=torch.uint8).reshape(5,16)
        scales = torch.arange(1,11,dtype=torch.float32).reshape(5,2).to(torch.float8_e4m3fn)
        table.load(packed, scales, 8, 10, 13, 0.3)
        self.assertEqual(table.nbytes, 3*(16+2))
        self.assertEqual(table.weight.dtype, torch.uint8)
        self.assertEqual(table.scales.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(table.weight, packed[2:5])
        torch.testing.assert_close(table.scales.float(), scales[2:5].float())
        self.assertEqual(table.global_scale, 0.3)
        table.finalize(10,13)

    def test_storage_rejects_invalid_layout_scale_and_incomplete_load(self):
        mod = runtime()
        for rows, dim in [(0,32),(2,17)]:
            with self.subTest(rows=rows,dim=dim), self.assertRaises(ValueError):
                mod.PackedPLEStorage(rows, dim, pin_memory=False)
        packed = torch.ones((2,16), dtype=torch.uint8)
        scales = torch.ones((2,2), dtype=torch.float8_e4m3fn)
        for weight, scale, global_scale in [(packed.float(),scales,1.), (packed,scales.float(),1.), (packed[:,:8],scales,1.), (packed,scales[:,:1],1.), (packed,scales,float('nan')), (packed,scales,0.)]:
            with self.subTest(dtype=weight.dtype,scale=global_scale), self.assertRaises(ValueError):
                mod.PackedPLEStorage(2,32,pin_memory=False).load(weight,scale,0,0,2,global_scale)
        table = mod.PackedPLEStorage(4,32,pin_memory=False)
        table.load(packed,scales,0,0,4,1.)
        with self.assertRaises(ValueError):
            table.load(packed,scales,2,0,4,2.)
        with self.assertRaises(ValueError):
            table.finalize(0,4)
        table.load(packed,scales,2,0,4,1.)
        table.finalize(0,4)
        with self.assertRaises(ValueError):
            table.load(packed,scales,2,0,4,1.)
            table.finalize(0,4)

    def test_storage_rejects_changed_destination_bounds(self):
        mod = runtime()
        packed = torch.ones((2,16),dtype=torch.uint8)
        scales = torch.ones((2,2),dtype=torch.float8_e4m3fn)
        table = mod.PackedPLEStorage(4,32,pin_memory=False)
        table.load(packed,scales,0,0,4,1.)
        with self.assertRaises(ValueError):
            table.finalize(0,2)
        with self.assertRaises(ValueError):
            table.load(packed,scales,2,2,6,1.)

    def test_synthetic_row_selection_and_optional_fp8_reference(self):
        mod = runtime()
        weights = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
        scales = torch.tensor([[.5, 2], [1, .25], [4, .125], [2, 1]], dtype=torch.float8_e4m3fn)
        ids = torch.tensor([3, 0, 2, 1, 3], dtype=torch.int64)
        lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
        codes = torch.stack((weights & 15, weights >> 4), dim=-1).reshape(4, 32).long()
        for reference in [False, True]:
            out = torch.empty((5, 32), dtype=torch.bfloat16)
            mod.gather_packed_kernel[(5,)](weights.data_ptr(), scales.data_ptr(), ids, out, .3, 32, 0, 4, reference, 32, enable_fp_fusion=False)
            if reference:
                expected = (lut[codes] * (scales.float().repeat_interleave(16, dim=1) * .3)).to(torch.float8_e4m3fn).bfloat16()
            else:
                expected = (lut[codes] * scales.float().repeat_interleave(16, dim=1) * .3).bfloat16()
            torch.testing.assert_close(out, expected[ids], rtol=0, atol=0)

    def test_all_finite_positive_e4m3_scales_and_fp8_rounding_ties(self):
        mod=runtime()
        weights=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]*2]*127,dtype=torch.uint8)
        scales=torch.arange(127,dtype=torch.uint8)[:,None].expand(127,2).contiguous().view(torch.float8_e4m3fn)
        ids=torch.arange(127,dtype=torch.int64)
        lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]).repeat(2)
        for reference in [False,True]:
            for global_scale in [1/32,0.3]:
                out=torch.empty((127,32),dtype=torch.bfloat16)
                mod.gather_packed_kernel[(127,)](weights.data_ptr(),scales.data_ptr(),ids,out,global_scale,32,0,127,reference,32,enable_fp_fusion=False)
                if reference:
                    expected=(lut*(scales.float().repeat_interleave(16,dim=1)*global_scale)).to(torch.float8_e4m3fn).bfloat16()
                else:
                    expected=(lut*scales.float().repeat_interleave(16,dim=1)*global_scale).bfloat16()
                try:
                    torch.testing.assert_close(out,expected,rtol=0,atol=0,equal_nan=True)
                except AssertionError:
                    mismatch = ~torch.isclose(out.float(),expected.float(),rtol=0,atol=0,equal_nan=True)
                    print('reference',reference,'global',global_scale,'bad actual',out[mismatch].float().tolist()[:8],'expected',expected[mismatch].float().tolist()[:8])
                    raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
