import importlib.util
from pathlib import Path

import numpy as np
import unittest
import tempfile

import gguf


spec = importlib.util.spec_from_file_location(
    "convert_lumen", Path(__file__).resolve().parents[2] / "scripts/convert_gemma4_to_lumen4exp.py")
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def make_source(path, missing_swa=False, legacy_nvfp4=False):
    writer = gguf.GGUFWriter(path, "gemma4")
    writer.add_block_count(30)
    writer.add_sliding_window_pattern(([True] * 5 + [False]) * 5)
    writer.add_shared_kv_layers(0)
    for layer in range(30):
        if missing_swa and layer == 28:
            continue
        writer.add_tensor(f"blk.{layer}.attn_k.weight", np.ones((64, 64), dtype=np.float32))
        writer.add_tensor(f"blk.{layer}.attn_k_norm.weight", np.ones(64, dtype=np.float32))
    values = np.linspace(-3, 3, 128, dtype=np.float32).reshape(2, 64)
    packed = gguf.quants.quantize(values, gguf.GGMLQuantizationType.NVFP4)
    if legacy_nvfp4:
        packed = packed[:, 4:].copy()
        writer.add_tensor_info("test.weight", (2, 64), np.dtype(np.float32), packed.nbytes,
                               raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        writer.kv_data[0]["general.quantization_version"].value = 2
    else:
        writer.add_tensor("test.weight", packed, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensors in writer.tensors:
        for info in list(tensors.values()):
            writer.write_tensor_data(packed if info.tensor is None else info.tensor)
    writer.close()


class TestLumen4Exp(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_conversion_preserves_payload(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                source, output = self.root / f"gemma-{legacy}.gguf", self.root / f"lumen-{legacy}.gguf"
                make_source(source, legacy_nvfp4=legacy)
                converter.convert(source, output)
                original, converted = gguf.GGUFReader(source), gguf.GGUFReader(output)
                self.assertEqual(converted.fields["general.architecture"].contents(), "lumen4exp")
                self.assertEqual(converted.fields["general.name"].contents(), "Lumen 3.8 Pulsar Next")
                self.assertEqual(converted.fields["lumen4exp.attention.shared_kv_layers"].contents(), 13)
                self.assertEqual(converted.fields["general.quantization_version"].contents(), 3)
                self.assertFalse(any(key.startswith("gemma4.") for key in converted.fields))
                for a, b in zip(original.tensors, converted.tensors):
                    self.assertEqual((a.name, a.tensor_type), (b.name, b.tensor_type))
                    np.testing.assert_array_equal(a.shape, b.shape)
                    if legacy and a.tensor_type == gguf.GGMLQuantizationType.NVFP4:
                        np.testing.assert_array_equal(a.data, b.data[:, 4:])
                        np.testing.assert_array_equal(b.data[:, :4].copy().view(np.float32), 1)
                    else:
                        np.testing.assert_array_equal(a.data, b.data)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    converter.convert(source, output)

    def test_missing_swa_weights_rejected(self):
        source = self.root / "gemma.gguf"
        make_source(source, missing_swa=True)
        with self.assertRaisesRegex(ValueError, "independent KV weights"):
            converter.convert(source, self.root / "lumen.gguf")

    def test_invalid_sharing_rejected(self):
        source = self.root / "gemma.gguf"
        make_source(source)
        for shared in (-1, 30, 29):
            with self.subTest(shared=shared), self.assertRaises(ValueError):
                converter.convert(source, self.root / "lumen.gguf", shared)

    def test_nvfp4_master_scale(self):
        values = np.array([[0] * 128, [1e-8] * 128, [1e8] * 128], dtype=np.float32)
        packed = gguf.quants.quantize(values, gguf.GGMLQuantizationType.NVFP4)
        self.assertEqual(packed.shape, (3, 76))
        scales = packed[:, :4].copy().view(np.float32).ravel()
        np.testing.assert_array_equal(scales, [1, np.float32(1e-8) / np.float32(2688), np.float32(1e8) / np.float32(2688)])
        restored = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.NVFP4)
        np.testing.assert_allclose(restored, values, rtol=1e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
