"""Parse the exact documented external argv with the installed SGLang parser.
No ServerArgs construction, checkpoint download, CUDA initialization or server.
"""
import argparse
import json
from test_quickstart import quickstart_argv
from sglang.srt.server_args import ServerArgs

words = quickstart_argv()
args = words[words.index('-m') + 2:]
parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
parsed = parser.parse_args(args)
assert parsed.model_path == 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4'
assert parsed.tp_size == 2
assert parsed.ple_offload_embedding
assert parsed.mm_enable_dp_encoder
assert not parsed.language_model_only
assert parsed.startup_weight_load_mode == 'serial'
assert parsed.context_length == 524288
assert parsed.quantization == 'modelopt_mixed'
assert json.loads(parsed.json_model_override_args)['text_config']['rope_parameters']['factor'] == 2.0
print(json.dumps({'actual_image_cli_parser': 'passed', 'tp_size': parsed.tp_size,
                  'context_length': parsed.context_length, 'gpu_boot': False}))
