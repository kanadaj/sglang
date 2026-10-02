import json
import pytest
from sglang.srt.entrypoints.openai.protocol import Tool, Function
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


def wire(name='example_function', values=None):
    values = values or {'value': '42'}
    return ('<tool_call><' + 'function=' + name + '>' + ''.join(
        '<' + 'parameter=' + k + '>' + v + '</parameter>' for k, v in values.items()
    ) + '</function></tool_call>')


def tool(schema=None):
    return Tool(type='function', function=Function(name='example_function', parameters=schema or {
        'type': 'object', 'properties': {'value': {'type': 'integer'}}}))


def parsed(text, tools, chunk):
    detector = Qwen3CoderDetector()
    if chunk is None:
        return [(c.name, json.loads(c.parameters)) for c in detector.detect_and_parse(text, tools).calls]
    by_index = {}
    for pos in range(0, len(text), chunk):
        for c in detector.parse_streaming_increment(text[pos:pos + chunk], tools).calls:
            state = by_index.setdefault(c.tool_index, ['', ''])
            if c.name:
                state[0] = c.name
            state[1] += c.parameters
    return [(name, json.loads(args)) for name, args in by_index.values()]


@pytest.mark.parametrize('keyword', ['anyOf', 'oneOf', 'allOf'])
@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
@pytest.mark.parametrize('kind,value,expected', [
    ('integer', '42', 42), ('boolean', 'true', True),
    ('object', '{"x": 1}', {'x': 1}), ('array', '[1, 2]', [1, 2])])
def test_composite_schema(keyword, chunk, kind, value, expected):
    schema = {keyword: [{'type': 'object', 'properties': {'value': {'type': kind}}}]}
    assert parsed(wire(values={'value': value}), [tool(schema)], chunk) == [('example_function', {'value': expected})]
