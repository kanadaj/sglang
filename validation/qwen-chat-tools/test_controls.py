import pytest
from test_schema import tool, wire, parsed
from sglang.srt.parser.reasoning_parser import BaseReasoningFormatDetector, Glm45Detector
from sglang.srt.function_call.utils import get_schema_properties


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_unknown_surrounding_prose_and_recovery(chunk):
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
    detector = Qwen3CoderDetector()
    raw = 'Before.' + wire('unknown') + 'ORCHID'
    if chunk is None:
        ret = detector.detect_and_parse(raw, [tool()])
        assert ret.calls == []
        assert ret.normal_text == 'Before.ORCHID'
    else:
        text = ''
        for pos in range(0, len(raw), chunk):
            ret = detector.parse_streaming_increment(raw[pos:pos+chunk], [tool()])
            assert ret.calls == []
            text += ret.normal_text
        assert text == 'Before.ORCHID'


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_unknown_forwarding_opt_in(monkeypatch, chunk):
    monkeypatch.setenv('SGLANG_FORWARD_UNKNOWN_TOOLS', 'true')
    assert parsed(wire('unknown'), [tool()], chunk) == [('unknown', {'value': '42'})]


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_no_declared_tools_preserves_previous_behavior(chunk):
    assert parsed(wire('unknown'), [], chunk) == [('unknown', {'value': '42'})]


def test_shared_defaults_and_glm_remain_immediate():
    for detector in (BaseReasoningFormatDetector('<think>', '</think>', tool_start_token='<tool_call>', force_reasoning=True), Glm45Detector()):
        assert detector.defer_tool_start is False
        ret = detector.parse_streaming_increment('<think>Planning.<tool_call>')
        assert ret.normal_text == '<tool_call>'
        assert ret.reasoning_text == 'Planning.'


def test_composite_helper_priority_and_nested_branches():
    assert get_schema_properties(None) == {}
    assert get_schema_properties({'properties': {}, 'anyOf': [{'properties': {'x': {'type': 'integer'}}}]}) == {}
    assert get_schema_properties({'anyOf': [{'allOf': [{'properties': {'x': {'type': 'integer'}}}]}, {'properties': {'x': {'type': 'string'}, 'y': {'type': 'boolean'}}}]}) == {'x': {'type': 'integer'}, 'y': {'type': 'boolean'}}
