"""Canonical Responses identities survive Qwen's flat function-name bridge."""

import asyncio
import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def member(name="get_marker", description="marker"):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "strict": True,
    }


def namespace(name="mcp__qwen_gate", members=None):
    return {"type": "namespace", "name": name, "tools": members or [member()]}


def request(**kwargs):
    return ResponsesRequest(
        **{
            "model": "x",
            "input": "Check the marker.",
            "tools": [namespace()],
            "store": False,
            **kwargs,
        }
    )


def serving(model="qwen3_8_flash_next"):
    s = make_serving()
    s.tokenizer_manager.model_config.hf_config.model_type = model
    s.reasoning_parser = "qwen3"
    s.tool_call_parser = "qwen3_coder"
    return s


def prepare(s, req):
    s._validate_media_content = Mock(return_value=None)
    s._process_messages = Mock(
        return_value=SimpleNamespace(prompt="test", prompt_ids=[1])
    )
    asyncio.run(s._make_request(req, None, s.tokenizer_manager.tokenizer))
    return s._process_messages.call_args.args[0]


@pytest.fixture(autouse=True)
def runtime():
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    yield
    reset_context()


@pytest.mark.parametrize(
    "model", ["qwen3_8_flash_next", "qwen3_8_flash_next_text", "qwen4_exp"]
)
def test_declaration_and_named_choice_use_same_flat_name(model):
    req = request(
        tool_choice={
            "type": "function",
            "namespace": "mcp__qwen_gate",
            "name": "get_marker",
        }
    )
    before = req.model_dump()
    chat = prepare(serving(model), req)
    assert chat.tools[0].function.name == "mcp__qwen_gate__get_marker"
    assert chat.tool_choice.function.name == "mcp__qwen_gate__get_marker"
    assert chat.tools[0].function.parameters == req.tools[0].tools[0]["parameters"]
    assert chat.tools[0].function.description == "marker"
    assert chat.tools[0].function.strict is True
    assert req.model_dump() == before


def test_non_qwen_retains_dotted_declarations_and_choice():
    chat = prepare(
        serving("llama"),
        request(
            tool_choice={
                "type": "function",
                "namespace": "mcp__qwen_gate",
                "name": "get_marker",
            }
        ),
    )
    assert chat.tools[0].function.name == "mcp__qwen_gate.get_marker"
    assert chat.tool_choice.function.name == "mcp__qwen_gate.get_marker"


@pytest.mark.parametrize("stored", [False, True])
def test_wire_replay_is_encoded_without_mutating_history(stored):
    s = serving()
    history = [
        {"role": "user", "content": "Check."},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan"}]},
        {"role": "assistant", "content": "Checking.", "phase": "commentary"},
        {
            "type": "function_call",
            "namespace": "mcp__qwen_gate",
            "name": "get_marker",
            "call_id": "call_1",
            "arguments": '{"value":"input"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "MARKER"},
    ]
    before = copy.deepcopy(history)
    if stored:
        s.msg_store["resp_prior"] = history[:-1]
        req = request(input=history[-1:], previous_response_id="resp_prior")
    else:
        req = request(input=history)
    messages = s._construct_input_messages(req)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    call = messages[1]["tool_calls"][0]
    assert call["function"]["name"] == "mcp__qwen_gate__get_marker"
    assert call["id"] == messages[2]["tool_call_id"] == "call_1"
    assert history == before


def native(name):
    return f"<tool_call><function={name}><parameter=value>MARKER</parameter></function></tool_call>"


@pytest.mark.parametrize(
    "wire_name", ["mcp__qwen_gate__get_marker", "mcp__qwen_gate.get_marker"]
)
@pytest.mark.parametrize("delivery", ["full", "coalesced", "characters", "seven"])
def test_full_and_fragmented_output_keep_canonical_identity(wire_name, delivery):
    s, req = serving(), request(stream=delivery != "full")
    raw = native(wire_name)
    if delivery == "full":
        items = [
            i.model_dump()
            for i in s._make_response_output_items(
                req, raw, Mock(), require_reasoning=False
            )
        ]
    else:
        chunks = (
            [raw]
            if delivery == "coalesced"
            else [
                raw[i : i + (1 if delivery == "characters" else 7)]
                for i in range(0, len(raw), 1 if delivery == "characters" else 7)
            ]
        )
        cumulative, engine = "", []
        for i, chunk in enumerate(chunks):
            cumulative += chunk
            engine.append(engine_chunk(cumulative, i + 1, finish=i == len(chunks) - 1))
        events = event_payloads(StreamFixture(s, req).run(engine))
        assert events[-1]["type"] == "response.completed"
        items = events[-1]["response"]["output"]
        for event in events:
            if event["type"] in (
                "response.output_item.added",
                "response.output_item.done",
            ):
                assert event["item"]["name"] == "get_marker"
                assert event["item"]["namespace"] == "mcp__qwen_gate"
    assert len(items) == 1
    assert (items[0]["name"], items[0].get("namespace")) == (
        "get_marker",
        "mcp__qwen_gate",
    )
    assert json.loads(items[0]["arguments"]) == {"value": "MARKER"}


def test_collisions_long_names_and_invalid_characters_are_exact_and_deterministic():
    declarations = [
        member("mcp__qwen_gate__get_marker", "ordinary"),
        {"type": "custom", "name": "other__inspect", "description": "custom"},
        namespace(),
        namespace("other", [member("inspect", "other")]),
        namespace("a__b", [member("c", "first")]),
        namespace("a", [member("b__c", "second")]),
        namespace("long" * 25, [member("member" * 15, "long")]),
        namespace("a.b", [member("c-d", "dot")]),
        namespace("a_b", [member("c-d", "underscore")]),
    ]
    req = request(tools=declarations)
    original = req.model_dump()
    s = serving()
    chat = prepare(s, req)
    names = [t.function.name for t in chat.tools]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)
    assert names[:2] == ["mcp__qwen_gate__get_marker", "other__inspect"]
    reversed_chat = prepare(s, request(tools=list(reversed(declarations))))
    assert {t.function.description: t.function.name for t in chat.tools} == {
        t.function.description: t.function.name for t in reversed_chat.tools
    }
    for declaration, tool in zip(declarations[2:], chat.tools[2:]):
        items = s._make_response_output_items(
            req, native(tool.function.name), Mock(), require_reasoning=False
        )
        assert len(items) == 1
        assert (items[0].namespace, items[0].name) == (
            declaration["name"],
            declaration["tools"][0]["name"],
        )
    ordinary = s._make_response_output_items(
        req, native(names[0]), Mock(), require_reasoning=False
    )[0].model_dump()
    assert ordinary["name"] == names[0] and ordinary.get("namespace") is None
    assert req.model_dump() == original


def test_ordinary_dotted_choice_and_explicit_namespace_choice_remain_distinct():
    tools = [namespace(), member("mcp__qwen_gate.get_marker", "ordinary")]
    for choice, expected in [
        (
            {"type": "function", "name": "mcp__qwen_gate.get_marker"},
            "mcp__qwen_gate.get_marker",
        ),
        (
            {"type": "function", "namespace": "mcp__qwen_gate", "name": "get_marker"},
            "mcp__qwen_gate__get_marker",
        ),
    ]:
        chat = prepare(serving(), request(tools=tools, tool_choice=choice))
        assert chat.tool_choice.function.name == expected


def test_unknown_suffix_is_not_guessed_and_maps_do_not_leak_between_requests():
    s = serving()
    unknown = "mcp__qwen_gate__get_marker_03f84705"
    item = s._make_response_output_items(
        request(), native(unknown), Mock(), require_reasoning=False
    )[0].model_dump()
    assert item["name"] == unknown and item.get("namespace") is None
    first = prepare(s, request()).tools[0].function.name
    second = prepare(s, request(tools=[member(first, "ordinary"), namespace()]))
    assert second.tools[1].function.name != first
    assert prepare(s, request()).tools[0].function.name == first


def test_undeclared_namespace_choice_is_rejected_before_generation():
    s = serving()
    for choice in [
        {"type": "function", "namespace": "missing", "name": "get_marker"},
        {"type": "function", "namespace": "mcp__qwen_gate", "name": "missing"},
        {"type": "function", "name": "mcp__qwen_gate__get_marker"},
    ]:
        response = asyncio.run(s.create_responses(request(tool_choice=choice)))
        assert response.status_code == 400
        assert b"tool_choice" in response.body
    s.tokenizer_manager.generate_request.assert_not_called()


@pytest.mark.parametrize(
    "name", ["mcp__qwen_gate.get_marker", "mcp__qwen_gate__get_marker"]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "kind,raw_value,expected",
    [
        ("integer", "7", 7),
        ("boolean", "false", False),
        ("array", "[1,2]", [1, 2]),
        ("object", '{"a":1}', {"a": 1}),
        ("string", "00123", "00123"),
    ],
)
def test_aliases_and_legacy_names_preserve_parameter_types(
    name, stream, kind, raw_value, expected
):
    tool = member()
    tool["parameters"]["properties"]["value"]["type"] = kind
    req = request(tools=[namespace(members=[tool])], stream=stream)
    s = serving()
    raw = native(name).replace("MARKER", raw_value)
    if stream:
        events = event_payloads(
            StreamFixture(s, req).run([engine_chunk(raw, finish=True)])
        )
        items = events[-1]["response"]["output"]
    else:
        items = [
            i.model_dump()
            for i in s._make_response_output_items(
                req, raw, Mock(), require_reasoning=False
            )
        ]
    assert json.loads(items[0]["arguments"])["value"] == expected
    assert (
        items[0]["name"] == "get_marker" and items[0]["namespace"] == "mcp__qwen_gate"
    )
