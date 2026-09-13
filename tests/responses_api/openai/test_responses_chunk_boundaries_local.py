"""Chunk delivery must not change Qwen Responses item order or payloads."""
import json
import re

import pytest
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

FUNCTION = {"type": "function", "name": "unused", "parameters": {
    "type": "object", "properties": {"value": {"type": "string"}},
}}
NAMESPACE = {"type": "namespace", "name": "lab", "tools": [FUNCTION]}
CUSTOM = {"type": "custom", "name": "unused"}


def call(name="unused", value=None, parameter="value"):
    body = "" if value is None else f"<parameter={parameter}>{value}</parameter>"
    return f"<tool_call><function={name}>{body}</function></tool_call>"


def message(text, phase="final_answer"):
    return ("message", phase, text)


def reasoning(text):
    return ("reasoning", text)


def function(value=None, namespace=None):
    return ("function_call", "unused", namespace, {} if value is None else {"value": value})


def semantic(item):
    if item["type"] == "message":
        return message("".join(c["text"] for c in item["content"]), item["phase"])
    if item["type"] == "reasoning":
        return reasoning("".join(c["text"] for c in item.get("content", [])))
    if item["type"] == "function_call":
        return (item["type"], item["name"], item.get("namespace"), json.loads(item["arguments"]))
    return (item["type"], item["name"], item["input"])


CASES = [
    ("tool_then_text", call() + "Final.", [function(), message("Final.")], [FUNCTION], False),
    ("both_sides", "Checking." + call() + "Final.",
     [message("Checking.", "commentary"), function(), message("Final.")], [FUNCTION], False),
    ("renewed", "Checking.<think>Again</think>Final.",
     [message("Checking.", "commentary"), reasoning("Again"), message("Final.")], [FUNCTION], False),
    ("forced_open", "Initial</think>Checking.<think>Again</think>" + call() + "Final.",
     [reasoning("Initial"), message("Checking.", "commentary"), reasoning("Again"),
      function(), message("Final.")], [FUNCTION], True),
    ("multiple_tools", call(value="α < β > γ") + "Between." + call(value="two") + "Final.",
     [function("α < β > γ"), message("Between.", "commentary"), function("two"), message("Final.")], [FUNCTION], False),
    ("namespace", "Checking." + call("lab.unused", "<b>café</b>") + "Final.",
     [message("Checking.", "commentary"), function("<b>café</b>", "lab"), message("Final.")], [NAMESPACE], False),
    ("custom", call(value="echo <b>café</b>", parameter="input") + "Final.",
     [("custom_tool_call", "unused", "echo <b>café</b>"), message("Final.")], [CUSTOM], False),
    ("literal_angles", "α < β > γ; <widget>text</widget> tail<",
     [message("α < β > γ; <widget>text</widget> tail<")], [], False),
    ("unfinished_marker", "Visible<thi", [message("Visible<thi")], [], False),
    ("unfinished_reasoning", "Checking.<think>Again", [message("Checking.", "commentary"), reasoning("Again")], [], False),
]


@pytest.mark.parametrize("case,raw,expected,tools,force", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("delivery", ["coalesced", "characters", "markup", "seven"])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_delivery_preserves_order_and_payload(case, raw, expected, tools, force, delivery, incremental, empty_terminal):
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        serving = make_serving()
        serving.tokenizer_manager.model_config.hf_config.model_type = "qwen3_8_flash_next"
        serving.reasoning_parser = "qwen3"
        serving.tool_call_parser = "qwen3_coder"
        serving.tokenizer_manager.server_args.incremental_streaming_output = incremental
        request = ResponsesRequest(model="x", input="test", stream=True, store=True, tools=tools)
        if delivery == "coalesced":
            parts = [raw]
        elif isinstance(delivery, int):
            parts = [raw[:delivery], raw[delivery:]]
        elif delivery == "characters":
            parts = list(raw)
        elif delivery == "markup":
            parts = [p for p in re.split(r"(?=<)|(?<=>)", raw) if p]
        else:
            parts = [raw[i:i + 7] for i in range(0, len(raw), 7)]
        if empty_terminal:
            parts.append("")
        cumulative = ""
        chunks = []
        for i, part in enumerate(parts):
            cumulative += part
            chunks.append(engine_chunk(part if incremental else cumulative, i + 1, finish=i == len(parts) - 1))
        events = event_payloads(StreamFixture(serving, request, require_reasoning=force).run(chunks))
        assert events[-1]["type"] == "response.completed"
        output = events[-1]["response"]["output"]
        assert [semantic(item) for item in output] == expected
        assert events[-1]["response"]["usage"]["output_tokens"] == len(parts)
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        assert done == output
        assert [x.model_dump() for x in serving.response_store[request.request_id].output] == output
        for e in events:
            if e["type"] == "response.output_item.added" and e["item"]["type"] == "message":
                assert e["item"]["phase"] is None
        assert sum(e["type"] == "response.completed" for e in events) == 1
    finally:
        reset_context()


@pytest.mark.parametrize("case,raw,expected,tools,force,cut", [
    (*case, cut) for case in CASES[:4] for cut in range(1, len(case[1]))
])
def test_every_single_split(case, raw, expected, tools, force, cut):
    test_delivery_preserves_order_and_payload(case, raw, expected, tools, force, cut, False, False)
