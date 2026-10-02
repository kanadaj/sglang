"""Compatibility controls independent of the splice attack."""
import pytest
from test_schema import parsed, tool, wire
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_unknown_only_preserves_prose_tail(chunk):
    text = 'before ' + wire('unknown') + ' after'
    detector = Qwen3CoderDetector()
    if chunk is None:
        results = [detector.detect_and_parse(text, [tool()])]
    else:
        results = [detector.parse_streaming_increment(text[pos:pos + chunk], [tool()])
                   for pos in range(0, len(text), chunk)]
    assert not any(result.calls for result in results)
    assert ''.join(result.normal_text for result in results) == 'before  after'


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
@pytest.mark.parametrize('tools', [None, []])
def test_detector_no_tools_keeps_compatibility(chunk, tools):
    assert parsed(wire(), tools, chunk) == [('example_function', {'value': '42'})]


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_forward_unknown_remains_opt_in(monkeypatch, chunk):
    monkeypatch.setenv('SGLANG_FORWARD_UNKNOWN_TOOLS', '1')
    assert parsed(wire('unknown') + wire(), [tool()], chunk) == [
        ('unknown', {'value': '42'}), ('example_function', {'value': 42})]
