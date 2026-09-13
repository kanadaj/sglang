import asyncio
import json
from unittest.mock import Mock

import pytest
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import RequestResponseMetadata, ResponsesRequest
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


FUNCTION = {"type": "function", "name": "unused", "parameters": {"type": "object"}}
NAMESPACE = {"type": "namespace", "name": "lab", "tools": [FUNCTION]}
CUSTOM = {"type": "custom", "name": "unused"}


@pytest.mark.parametrize("tools,choice", [
    ([], "auto"), ([FUNCTION], "none"), ([FUNCTION], "auto"),
    ([NAMESPACE], "auto"), ([CUSTOM], "auto"),
])
@pytest.mark.parametrize("qwen_parsers", [False, True])
@pytest.mark.parametrize("ending", ["complete", "cancel", "error"])
def test_text_arrives_before_generation_finishes(tools, choice, ending, qwen_parsers):
    async def check():
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        if qwen_parsers:
            serving.tokenizer_manager.model_config.hf_config.model_type = "qwen3_8_flash_next"
            serving.reasoning_parser = "qwen3"
            serving.tool_call_parser = "qwen3_coder"
        request = ResponsesRequest(
            model="x", input="Tell a story", stream=True, store=True,
            tool_choice=choice, tools=tools,
        )
        release = asyncio.Event()

        async def generate():
            yield engine_chunk("Once upon a time", 5)
            await release.wait()
            if ending == "error":
                raise RuntimeError("synthetic generation failure")
            yield engine_chunk("Once upon a time. The end.", 9, finish=True)

        stream = serving.responses_stream_generator_non_harmony(
            request, sampling_params={}, result_generator=generate(), model_name="x",
            tokenizer=Mock(), request_metadata=RequestResponseMetadata(request_id=request.request_id),
            require_reasoning=False,
        )
        events = []

        async def first_text():
            async for event in stream:
                payload = json.loads(event.split("data: ", 1)[1])
                events.append(payload)
                if payload["type"] == "response.output_text.delta":
                    return payload["delta"]
            raise AssertionError("Stream ended without text")

        try:
            # The producer cannot finish until the consumer receives its first text.
            assert await asyncio.wait_for(first_text(), timeout=1) == "Once upon a time"
            added = next(e["item"] for e in events if e["type"] == "response.output_item.added")
            assert added["phase"] == (None if qwen_parsers or (tools and choice != "none") else "final_answer")
            assert request.request_id not in serving.response_store
            if ending == "cancel":
                await stream.aclose()
                assert request.request_id not in serving.response_store
                assert request.request_id not in serving.msg_store
                assert not any(e["type"] == "response.output_item.done" for e in events)
                return
            release.set()
            async for event in stream:
                events.append(json.loads(event.split("data: ", 1)[1]))
            assert [e["sequence_number"] for e in events] == list(range(len(events)))
            terminal = "response.failed" if ending == "error" else "response.completed"
            assert events[-1]["type"] == terminal
            assert sum(e["type"] == terminal for e in events) == 1
            if ending == "error":
                assert events[-1]["response"]["error"]["message"] == "synthetic generation failure"
                assert request.request_id not in serving.response_store
            else:
                done = next(e["item"] for e in events if e["type"] == "response.output_item.done")
                assert done["phase"] == "final_answer"
                assert events[-1]["response"]["output"] == [done]
                assert serving.response_store[request.request_id].output[0].model_dump() == done
                assert serving.msg_store[request.request_id][-1]["phase"] == "final_answer"
        finally:
            release.set()
            await stream.aclose()

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        asyncio.run(check())
    finally:
        reset_context()


@pytest.mark.parametrize("suffix,expected_kinds,expected_phases", [
    ("</think><tool_call><function=unused></function></tool_call>", ["message", "reasoning", "function_call"], ["commentary"]),
    ("</think>Final answer.", ["message", "reasoning", "message"], ["commentary", "final_answer"]),
    ("", ["message", "reasoning"], ["commentary"]),
])
def test_text_displaced_by_reasoning_closes_as_commentary(suffix, expected_kinds, expected_phases):
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    try:
        serving = make_serving()
        serving.reasoning_parser = "qwen3"
        serving.tool_call_parser = "qwen3_coder"
        request = ResponsesRequest(model="x", input="test", stream=True, store=True, tools=[FUNCTION])
        texts = ["Let me check.", "Let me check.<think>Working", "Let me check.<think>Working" + suffix]
        events = event_payloads(StreamFixture(serving, request).run([
            engine_chunk(text, i + 1, finish=i == len(texts) - 1) for i, text in enumerate(texts)
        ]))
        output = events[-1]["response"]["output"]
        assert [item["type"] for item in output] == expected_kinds
        assert [item["phase"] for item in output if item["type"] == "message"] == expected_phases
        for e in events:
            if e["type"] == "response.output_item.added" and e["item"]["type"] == "message":
                assert e["item"]["phase"] is None
            if e["type"] == "response.output_item.done":
                assert e["item"] == output[e["output_index"]]
        done_pos = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.done")
        reasoning_pos = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.added" and e["item"]["type"] == "reasoning")
        assert done_pos < reasoning_pos
        assert [item.model_dump() for item in serving.response_store[request.request_id].output] == output
        assert [item["phase"] for item in serving.msg_store[request.request_id] if item.get("phase")] == expected_phases
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
    finally:
        reset_context()
