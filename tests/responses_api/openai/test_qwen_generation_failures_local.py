"""CPU regressions for generated tool syntax and invalid-token failures."""

from utils import StreamFixture, engine_chunk, event_payloads, make_serving

import asyncio
from array import array
from copy import deepcopy
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
    Tool,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import reset_context

TOOL = {
    "name": "inspect",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "count": {"type": "integer"}},
    },
}
CHAT_TOOLS = [{"type": "function", "function": TOOL}]
RESPONSE_TOOLS = [{"type": "function", **TOOL}]
CALL = "<tool_call><function=inspect></function></tool_call>"
STOP = {"type": "stop", "matched": 248046}
LENGTH = {"type": "length", "length": 10}
MALFORMED = [
    '<tool_call>{"name":"inspect","arguments":{}}</tool_call>',
    "<tool_call></tool_call>",
    "<tool_call>discarded text" + CALL,
    "<tool_call><unexpected/></tool_call>",
    "<tool_call><function=></function></tool_call>",
    "<tool_call><parameter=command>x</parameter></tool_call>",
    "<tool_call><function=inspect>discarded text</function></tool_call>",
    "<tool_call><function=inspect><function=inspect></function></tool_call>",
    "<tool_call><function=inspect></tool_call>",
]
UNFINISHED = [
    "<tool_call>",
    "<tool_call><function=ins",
    "<tool_call><function=inspect>",
    "<tool_call><function=inspect><parameter=com",
    "<tool_call><function=inspect><parameter=command>pwd",
    "<tool_call><function=inspect></function>",
]


@pytest.fixture(autouse=True)
def context():
    reset_context()
    yield
    reset_context()


def parser():
    return FunctionCallParser([Tool(**t) for t in CHAT_TOOLS], "qwen3_coder")


def scheduled(ids, cap=20, eos=True, accepted=None):
    req = Req.__new__(Req)
    req.output_ids = array("q", ids)
    req.vocab_size = 100
    req.sampling_params = SimpleNamespace(
        stop_token_ids={2} if eos else set(),
        max_new_tokens=cap,
        stop_strs=[],
        stop_regex_strs=[],
        ignore_eos=False,
    )
    req.eos_token_ids = {2} if eos else set()
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.grammar = None
    req.tokenizer = None
    req.update_finish_state(new_accepted_len=accepted or len(ids))
    return req


@pytest.mark.parametrize("bad_id", [-1, 100, 123456])
@pytest.mark.parametrize("cap", [1, 2, 20])
@pytest.mark.parametrize("eos", [False, True])
def test_invalid_token_is_failure_even_past_length_cap(bad_id, cap, eos):
    req = scheduled([5, 6, bad_id, 9], cap, eos)
    finish = req.finished_reason.to_json()
    assert finish["type"] == "abort"
    assert finish["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert finish["err_type"] == "InvalidTokenError"
    assert list(req.output_ids_through_stop) == [5, 6][:cap]


def test_invalid_first_token_never_reaches_decode():
    req = scheduled([-1], eos=False)
    assert list(req.output_ids_through_stop) == []
    assert req.finished_reason.to_json()["type"] == "abort"


@pytest.mark.parametrize(
    "ids,cap,expected",
    [
        ([5, 2], 20, "stop"),
        ([5, 6, 2], 1, "length"),
        ([5, 6], 2, "length"),
        ([5, 6], 20, None),
    ],
)
def test_ordinary_scheduler_finishes(ids, cap, expected):
    req = scheduled(ids, cap)
    actual = req.finished_reason.to_json()["type"] if req.finished_reason else None
    assert actual == expected


@pytest.mark.parametrize("text", MALFORMED + UNFINISHED)
def test_non_stream_parser_rejects_lost_tool_syntax(text):
    with pytest.raises(RuntimeError):
        parser().parse_non_stream(text)


@pytest.mark.parametrize("text", MALFORMED + UNFINISHED)
@pytest.mark.parametrize("split", [False, True])
def test_stream_parser_rejects_lost_tool_syntax(text, split):
    p = parser()
    with pytest.raises(RuntimeError):
        for chunk in list(text) if split else [text]:
            p.parse_stream_chunk(chunk)
        p.parse_stream_end()


@pytest.mark.parametrize(
    "text",
    [
        CALL,
        "\n<tool_call>\n<function=inspect>\n<parameter=command>\nif a < b: print('x')\n</parameter>\n"
        "<parameter=count>3</parameter>\n</function>\n</tool_call>\n",
        "<tool_call><function=inspect><parameter=command>pwd</function></tool_call>",
        CALL + CALL,
        "<tool_call><function=inspect></function><function=inspect></function></tool_call>",
    ],
)
def test_valid_tool_arguments_are_independent_of_split(text):
    import json

    _, expected = parser().parse_non_stream(text)
    expected_calls = [(c.name, json.loads(c.parameters)) for c in expected]
    for split_at in range(len(text) + 1):
        p = parser()
        calls = {}
        for chunk in [text[:split_at], text[split_at:]]:
            _, parts = p.parse_stream_chunk(chunk)
            for part in parts:
                target = calls.setdefault(part.tool_index, {"name": None, "args": ""})
                if part.name:
                    target["name"] = part.name
                target["args"] += part.parameters or ""
        p.parse_stream_end()
        assert [
            (c["name"], json.loads(c["args"])) for c in calls.values()
        ] == expected_calls


def test_partial_marker_outside_tool_is_text():
    p = parser()
    head, _ = p.parse_stream_chunk("Literal <")
    tail, calls = p.parse_stream_end()
    assert head + tail == "Literal <"
    assert not calls


def serving():
    s = make_serving()
    s.tokenizer_manager.model_config.hf_config.model_type = "qwen3_8_flash_next"
    s.reasoning_parser = "qwen3"
    s.tool_call_parser = "qwen3_coder"
    s.tokenizer_manager.abort_request = Mock()
    s.tokenizer_manager.request_logger = SimpleNamespace(log_requests=False)
    return s


def chunks(parts, finish, incremental=False):
    result, accumulated = [], ""
    for n, part in enumerate(parts):
        accumulated += part
        item = engine_chunk(part if incremental else accumulated, n + 1)
        item["index"] = 0
        item["meta_info"]["weight_version"] = "fixture"
        item["meta_info"]["finish_reason"] = finish if n == len(parts) - 1 else None
        result.append(item)
    return result


async def generated(items, is_stream):
    for item in items:
        if item["meta_info"]["finish_reason"]:
            manager = SimpleNamespace(rid_to_state={}, enable_lora=False)
            state = SimpleNamespace(obj=SimpleNamespace(rid="rid"))
            await TokenizerManager._handle_abort_finish_reason(
                manager, item, state, is_stream
            )
        yield deepcopy(item)


def requests(api, stream):
    if api == "responses":
        return ResponsesRequest(
            model="x",
            input="Inspect.",
            stream=stream,
            store=True,
            tools=RESPONSE_TOOLS,
            reasoning={"effort": "xhigh"},
        )
    return ChatCompletionRequest(
        model="x",
        messages=[{"role": "user", "content": "Inspect."}],
        stream=stream,
        tools=CHAT_TOOLS,
        reasoning_effort="xhigh",
    )


def drive(api, parts, finish, stream, incremental=False):
    import json

    s = serving()
    s.tokenizer_manager.server_args.incremental_streaming_output = incremental
    request = requests(api, stream)
    items = chunks(parts, finish, incremental)
    s.tokenizer_manager.generate_request.side_effect = lambda *a, **kw: generated(
        items, stream
    )
    if api == "responses" and stream:
        events = event_payloads(
            StreamFixture(s, request, require_reasoning=True).run(items)
        )
        return events, s, request

    async def run():
        if api == "responses":
            ctx = SimpleContext()

            async def gen():
                async for item in generated(items, False):
                    ctx.append_output(item)
                    yield ctx

            return await s.responses_full_generator(
                request,
                {},
                gen(),
                ctx,
                "x",
                s.tokenizer_manager.tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=True,
            )
        adapted = SimpleNamespace(stream=stream, rid="rid")
        if api == "chat":
            if stream:
                return [
                    line
                    async for line in s._generate_chat_stream(adapted, request, None)
                ]
            s._convert_to_internal_request = Mock(return_value=(adapted, request))
            return await s.handle_request(request, None)
        messages = AnthropicMessagesRequest(
            model="x",
            messages=[{"role": "user", "content": "Inspect."}],
            max_tokens=64,
            stream=stream,
        )
        handler = AnthropicServing(s)
        if stream:
            return [
                line
                async for line in handler._generate_anthropic_stream(
                    adapted, request, messages, None
                )
            ]
        s._convert_to_internal_request = Mock(return_value=(adapted, request))
        return await handler._handle_non_streaming(request, messages, None)

    result = asyncio.run(run())
    if api != "responses":
        s.tokenizer_manager.generate_request.assert_called_once()
    if stream:
        result = [
            json.loads(line[6:])
            for frame in result
            for line in frame.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    return result, s, request


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad", [MALFORMED[0], UNFINISHED[1], UNFINISHED[4]])
@pytest.mark.parametrize("think_close", ["</think>", "</think>\n\n", ""])
def test_three_api_malformed_generation_is_visible_failure(
    api, stream, bad, think_close
):
    result, s, request = drive(api, ["Planning." + think_close + bad], STOP, stream)
    if not stream:
        assert result.status_code == 500
    elif api == "responses":
        assert result[-1]["type"] == "response.failed"
        assert s.response_store[request.request_id].status == "failed"
        assert not any(e["type"] == "response.completed" for e in result)
        assert not any(
            e["type"] == "response.output_item.done"
            and e["item"]["type"] == "function_call"
            for e in result
        )
    elif api == "chat":
        assert result[-1]["error"]["code"] == 500
        assert not any(
            c.get("finish_reason") in ("stop", "tool_calls")
            for e in result
            for c in e.get("choices", [])
        )
    else:
        assert any(
            e["type"] == "error" and e["error"]["type"] == "api_error" for e in result
        )


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_token_fault_reaches_each_api(api, stream):
    finish = scheduled([5, -1]).finished_reason.to_json()
    result, _, _ = drive(api, ["Planning only."], finish, stream)
    if not stream:
        assert result.status_code == 500
    elif api == "responses":
        assert result[-1]["type"] == "response.failed"
        assert "invalid token ID" in result[-1]["response"]["error"]["message"]
    elif api == "chat":
        assert result[-1]["error"]["code"] == 500
    else:
        assert any(e["type"] == "error" for e in result)


@pytest.mark.parametrize("incremental", [False, True])
def test_mid_stream_failure_aborts_only_its_generation(incremental):
    result, s, request = drive(
        "responses",
        ["Planning.</think>", "<tool_call>bad", "never consumed"],
        STOP,
        True,
        incremental,
    )
    assert result[-1]["type"] == "response.failed"
    s.tokenizer_manager.abort_request.assert_called_once_with(rid=request.request_id)
    assert s.response_store[request.request_id].status == "failed"


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "ending", ["Planning only.", "Planning.</think>\n\n", "Planning.</think>Answer."]
)
def test_normal_eos_behavior_is_unchanged(api, stream, ending):
    result, _, _ = drive(api, [ending], STOP, stream)
    if not stream:
        assert getattr(result, "status_code", 200) == 200
    elif api == "responses":
        assert result[-1]["type"] == "response.completed"
    elif api == "chat":
        assert any(
            c.get("finish_reason") == "stop"
            for e in result
            for c in e.get("choices", [])
        )
    else:
        assert not any(e["type"] == "error" for e in result)


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_length_cutoff_is_incomplete_without_executable_partial_call(api, stream):
    import json

    text = "Planning.</think>Before tool.<tool_call><function=inspect><parameter=command>pw"
    result, _, _ = drive(api, [text], LENGTH, stream)
    if api == "responses":
        final = result[-1]["response"] if stream else result.model_dump()
        assert final["status"] == "incomplete"
        assert final["incomplete_details"] == {"reason": "max_output_tokens"}
        assert not any(x["type"] == "function_call" for x in final["output"])
        if stream:
            assert not any(
                e["type"] == "response.output_item.done"
                and e["item"]["type"] == "function_call"
                for e in result
            )
    elif api == "chat":
        if stream:
            assert any(
                c.get("finish_reason") == "length"
                for e in result
                for c in e.get("choices", [])
            )
            assert not any("error" in e for e in result)
        else:
            assert result.choices[0].finish_reason == "length"
            assert not result.choices[0].message.tool_calls
    elif stream:
        assert any(
            e["type"] == "message_delta" and e["delta"]["stop_reason"] == "max_tokens"
            for e in result
        )
        assert not any(e["type"] == "error" for e in result)
    else:
        assert json.loads(result.body)["stop_reason"] == "max_tokens"


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
@pytest.mark.parametrize("incremental", [False, True])
def test_valid_split_tool_still_completes(api, incremental):
    result, _, _ = drive(
        api,
        [
            "Planning.</think>",
            "<tool_",
            "call><fun",
            "ction=inspect>",
            "</fun",
            "ction></tool_call>",
        ],
        STOP,
        True,
        incremental,
    )
    if api == "responses":
        assert result[-1]["type"] == "response.completed"
        assert any(
            e["type"] == "response.output_item.done"
            and e["item"]["type"] == "function_call"
            for e in result
        )
    elif api == "chat":
        assert any(
            c.get("finish_reason") == "tool_calls"
            for e in result
            for c in e.get("choices", [])
        )
    else:
        assert any(
            e["type"] == "message_delta" and e["delta"]["stop_reason"] == "tool_use"
            for e in result
        )


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("ending", ["invalid_token", "length", "malformed"])
def test_partial_call_never_gets_done_on_failed_or_truncated_stream(
    incremental, ending
):
    finish = (
        scheduled([5, -1]).finished_reason.to_json()
        if ending == "invalid_token"
        else LENGTH if ending == "length" else STOP
    )
    parts = ["Planning.</think><tool_call><function=inspect>", "<parameter=command>pw"]
    events, _, _ = drive("responses", parts, finish, True, incremental)
    assert any(
        e["type"] == "response.output_item.added"
        and e["item"]["type"] == "function_call"
        for e in events
    )
    assert not any(
        e["type"] == "response.output_item.done"
        and e["item"]["type"] == "function_call"
        for e in events
    )
    assert events[-1]["type"] == (
        "response.incomplete" if ending == "length" else "response.failed"
    )


@pytest.mark.parametrize("api", ["responses", "chat", "messages"])
def test_integer_abort_status_survives_metadata_serialization(api):
    finish = scheduled([5, -1]).finished_reason.to_json()
    finish["status_code"] = int(finish["status_code"])
    events, _, _ = drive(api, ["Planning only."], finish, True)
    if api == "responses":
        assert events[-1]["type"] == "response.failed"
    elif api == "chat":
        assert events[-1]["error"]["code"] == 500
    else:
        assert any(e["type"] == "error" for e in events)


def test_unknown_well_formed_tool_is_not_a_parser_error():
    _, calls = parser().parse_non_stream(
        CALL.replace("function=inspect", "function=unknown_namespace.unknown_tool")
    )
    assert len(calls) == 1
    assert calls[0].name == "unknown_namespace.unknown_tool"


@pytest.mark.parametrize("is_stream", [False, True])
def test_invalid_token_abort_cleans_tokenizer_state(is_stream):
    from fastapi import HTTPException

    item = chunks([""], scheduled([-1]).finished_reason.to_json())[0]
    manager = SimpleNamespace(rid_to_state={"rid": object()}, enable_lora=False)
    state = SimpleNamespace(obj=SimpleNamespace(rid="rid"))

    async def run():
        return await TokenizerManager._handle_abort_finish_reason(
            manager, item, state, is_stream
        )

    if is_stream:
        assert asyncio.run(run()) is item
    else:
        with pytest.raises(HTTPException) as raised:
            asyncio.run(run())
        assert raised.value.status_code == 500
    assert not manager.rid_to_state
