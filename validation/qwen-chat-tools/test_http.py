"""Real ASGI adapters with injected CPU generation, no model/GPU required."""
import json
import pytest
import runtime_responses_compat as compat
from test_schema import wire


@pytest.fixture
def http():
    compat.MockHTTPTest.setUpClass()
    case = compat.MockHTTPTest()
    case.setUp()
    old_chat = getattr(case.app.state, 'openai_serving_chat', None)
    case.app.state.openai_serving_chat = case.chat
    case.chat.tokenizer_manager.server_args.context_length = 32768
    case.chat.tokenizer_manager.request_logger.log_requests = False
    case.chat.tokenizer_manager.create_abort_task.return_value = None
    case.chat.reasoning_parser = case.serving.reasoning_parser = 'qwen3'
    case.chat.tool_call_parser = case.serving.tool_call_parser = 'qwen3_coder'
    yield case
    case.app.state.openai_serving_chat = old_chat
    case.tearDown()


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('chunk', [1, 7, 10000])
@pytest.mark.parametrize('scenario', ['quoted', 'unclosed', 'valid', 'unknown', 'anyOf', 'oneOf', 'allOf'])
def test_http_boundaries_and_final_flush(http, api, chunk, scenario):
    composite = scenario in ('anyOf', 'oneOf', 'allOf')
    values = {'count': '42', 'flag': 'true', 'payload': '{"x": 1}', 'items': '[1, 2]'}
    raw = wire(values=values) if composite else {'quoted': '<think>Example: ' + wire() + '</think>ORCHID',
           'unclosed': '<think>Planning.' + wire(),
           'valid': '<think>Planning.</think>' + wire(),
           'unknown': wire('unknown') + 'ORCHID'}[scenario]
    async def generate(request, *args, **kwargs):
        sizes = list(range(chunk, len(raw), chunk)) + [len(raw)] if request.stream else [len(raw)]
        for size in sizes:
            yield {'text': raw[:size], 'output_ids': [1] * size, 'meta_info': {
                'id': 'fixture', 'weight_version': 'fixture', 'prompt_tokens': 10, 'completion_tokens': size,
                'finish_reason': {'type': 'stop'} if size == len(raw) else None}}
    http.serving.tokenizer_manager.generate_request = generate
    tools = [{'type': 'function', 'name': 'example_function', 'parameters': {
        'type': 'object', 'properties': {'value': {'type': 'integer'}}}}]
    if composite:
        tools[0]['parameters'] = {scenario: [{'type': 'object', 'properties': {
            'count': {'type': 'integer'}, 'flag': {'type': 'boolean'},
            'payload': {'type': 'object'}, 'items': {'type': 'array'}}}]}
    results = []
    for stream in (False, True):
        if api == 'responses':
            response = http.send(stream=stream, tools=tools, tool_choice='auto')
            assert response.status_code == 200
            body = next(e['response'] for e in http.events(response) if e['type'] == 'response.completed') if stream else response.json()
            calls = [(i['name'], json.loads(i['arguments'])) for i in body['output'] if i['type'] == 'function_call']
            normal = ''.join(p['text'] for i in body['output'] if i['type'] == 'message' for p in i['content'])
            reasoning = ''.join(p['text'] for i in body['output'] if i['type'] == 'reasoning' for p in i.get('content', []))
        else:
            response = http.client.post('/v1/chat/completions', json={
                'model': 'fixture-qwen', 'messages': [{'role': 'user', 'content': 'Hi'}],
                'tools': [{'type': 'function', 'function': t} for t in tools],
                'tool_choice': 'auto', 'stream': stream, 'max_tokens': 256})
            assert response.status_code == 200
            if stream:
                events = [json.loads(l[6:]) for l in response.text.splitlines() if l.startswith('data: ') and l != 'data: [DONE]']
                deltas = [e['choices'][0]['delta'] for e in events if e.get('choices')]
                normal = ''.join(d.get('content') or '' for d in deltas)
                reasoning = ''.join(d.get('reasoning_content') or '' for d in deltas)
                states = {}
                for d in deltas:
                    for c in d.get('tool_calls') or []:
                        st = states.setdefault(c['index'], ['', ''])
                        st[0] = c['function'].get('name') or st[0]
                        st[1] += c['function'].get('arguments') or ''
                calls = [(name, json.loads(args)) for name, args in states.values()]
            else:
                msg = response.json()['choices'][0]['message']
                normal, reasoning = msg.get('content') or '', msg.get('reasoning_content') or ''
                calls = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in msg.get('tool_calls') or []]
        expected = [('example_function', {'count': 42, 'flag': True, 'payload': {'x': 1}, 'items': [1, 2]})] if composite else [('example_function', {'value': 42})] if scenario in ('valid', 'unclosed') else []
        assert calls == expected
        if scenario == 'quoted':
            assert normal == 'ORCHID'
            assert reasoning == 'Example: ' + wire()
        results.append((normal, reasoning, calls))
    assert results[0] == results[1]
