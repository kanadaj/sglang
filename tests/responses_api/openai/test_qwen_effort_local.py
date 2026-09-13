"""Real Qwen template + real API conversion/render path; no model/GPU required."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from utils import make_serving  # CPU bootstrap first
import pytest
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.parser.template_detection import ReasoningToggleConfig


@pytest.fixture(scope="module")
def serving():
    s = make_serving()
    s.tokenizer_manager.model_config.hf_config.model_type = "qwen3_8_flash_next"
    snapshot = Path(os.environ['QWEN_REPLAY_MODEL_PATH'])
    s.tokenizer_manager.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    s.template_manager.chat_template_name = None
    s.template_manager.jinja_template_content_format = 'string'
    s.template_manager.reasoning_config = ReasoningToggleConfig(toggle_param='enable_thinking', default_enabled=True)
    s.default_chat_template_kwargs = {'reasoning_effort': 'medium'}
    s.reasoning_parser = 'qwen3'
    s.tool_call_parser = None
    return s


def render(s, protocol, effort):
    if protocol == 'responses':
        r = ResponsesRequest(model='x', input='Say ACK.', store=False,
                             reasoning={'effort': effort} if effort else None)
        return asyncio.run(s._make_request(r, None, s.tokenizer_manager.tokenizer))[-1]
    if protocol == 'messages':
        r = AnthropicMessagesRequest(model='x', messages=[{'role':'user','content':'Say ACK.'}], max_tokens=32,
            thinking={'type':'disabled'} if effort == 'none' else {'type':'adaptive'},
            output_config={'effort':effort} if effort and effort != 'none' else None)
        r = AnthropicServing(s)._convert_to_chat_completion_request(r)
    else:
        r = ChatCompletionRequest(model='x', messages=[{'role':'user','content':'Say ACK.'}], reasoning_effort=effort)
    return s._process_messages(r, False)


@pytest.mark.parametrize('protocol', ['chat','responses','messages'])
@pytest.mark.parametrize('effort', [None,'none','minimal','low','medium','high','xhigh','max'])
def test_actual_three_protocol_render(serving, protocol, effort):
    baseline = deepcopy(serving.default_chat_template_kwargs)
    result = render(serving, protocol, effort)
    text = result.prompt or serving.tokenizer_manager.tokenizer.decode(result.prompt_ids)
    canonical = {'minimal':'low','high':'xhigh','max':'xhigh'}.get(effort,effort)
    assert result.require_reasoning == (effort != 'none')
    if effort == 'none':
        assert text.endswith('<think>\n\n</think>\n\n')
    else:
        assert text.endswith('<think>\n')
    assert ('Reasoning effort is set to xhigh.' in text) == (canonical == 'xhigh')
    assert ('Reasoning effort is set to low.' in text) == (canonical == 'low')
    assert serving.default_chat_template_kwargs == baseline


def test_concurrent_cross_protocol_no_leakage(serving):
    cases = [(p,e) for _ in range(3) for p in ['chat','responses','messages'] for e in ['none','max',None,'low']]
    def check(case): test_actual_three_protocol_render(serving,*case)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(check,cases))


def test_explicit_template_effort_beats_top_and_default(serving):
    r=ChatCompletionRequest(model='x',messages=[{'role':'user','content':'Say ACK.'}],
        reasoning_effort='low',chat_template_kwargs={'reasoning_effort':'max'})
    out=serving._process_messages(r,False)
    text = out.prompt or serving.tokenizer_manager.tokenizer.decode(out.prompt_ids)
    assert 'Reasoning effort is set to xhigh.' in text
    assert serving.default_chat_template_kwargs == {'reasoning_effort':'medium'}


def test_disabled_anthropic_survives_max_effort(serving):
    r=AnthropicMessagesRequest(model='x',messages=[{'role':'user','content':'Say ACK.'}],max_tokens=32,
        thinking={'type':'disabled'},output_config={'effort':'max'})
    out=serving._process_messages(AnthropicServing(serving)._convert_to_chat_completion_request(r),False)
    assert not out.require_reasoning
    text = out.prompt or serving.tokenizer_manager.tokenizer.decode(out.prompt_ids)
    assert text.endswith('<think>\n\n</think>\n\n')
