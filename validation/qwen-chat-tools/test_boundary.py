import pytest
from test_schema import wire, tool, parsed
from sglang.srt.parser.reasoning_parser import ReasoningParser


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_undeclared_name_suppressed_and_valid_recovery(chunk):
    assert parsed(wire('unknown') + wire(), [tool()], chunk) == [('example_function', {'value': 42})]


@pytest.mark.parametrize('chunk', [1, 7, 10000])
@pytest.mark.parametrize('stream_reasoning', [True, False])
@pytest.mark.parametrize('closed', [True, False])
def test_reasoning_tool_boundary_parity(chunk, stream_reasoning, closed):
    body = 'Example quoted: ' + wire()
    text = '<think>' + body + ('</think>ORCHID' if closed else '')
    expected = ReasoningParser('qwen3', stream_reasoning=stream_reasoning).parse_non_stream(text)
    parser = ReasoningParser('qwen3', stream_reasoning=stream_reasoning)
    reasoning, normal = '', ''
    for pos in range(0, len(text), chunk):
        r, n = parser.parse_stream_chunk(text[pos:pos+chunk])
        reasoning += r or ''
        normal += n or ''
    r, n = parser.parse_stream_end()
    reasoning += r or ''
    normal += n or ''
    assert (reasoning, normal) == expected
    assert parser.parse_stream_end() == ('', '')
    if closed:
        assert parsed(normal, [tool()], chunk) == []
        assert normal == 'ORCHID'
    else:
        assert parsed(normal, [tool()], chunk) == [('example_function', {'value': 42})]
