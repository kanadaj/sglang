"""Responses replay must preserve native Qwen assistant turns and item order."""
import copy
import json
import os

import pytest
from utils import make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    normalize_assistant_tool_call_arguments,
)


def reasoning(text="PLAN"):
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def message(text="CHECKING", phase="commentary"):
    return {"role": "assistant", "content": text, "phase": phase}


def call(name="inspect", call_id="call_1", **extra):
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": '{"value":"MARKER"}', **extra}


def tool_result(call_id="call_1"):
    return {"type": "function_call_output", "call_id": call_id, "output": "HEALTHY"}


def construct(items, model="qwen3_8_flash_next", stored=False):
    serving = make_serving()
    serving.tokenizer_manager.model_config.hf_config.model_type = model
    prefix = [{"role": "user", "content": "Inspect status and give a final report."}]
    if stored:
        serving.msg_store["resp_prior"] = copy.deepcopy(prefix + items[:-1])
        request = ResponsesRequest(model="x", input=[items[-1]], previous_response_id="resp_prior")
    else:
        request = ResponsesRequest(model="x", input=copy.deepcopy(prefix + items))
    before = copy.deepcopy(request.model_dump())
    stored_before = copy.deepcopy(serving.msg_store)
    output = serving._construct_input_messages(request)
    assert request.model_dump() == before
    assert serving.msg_store == stored_before
    return output


@pytest.mark.parametrize("model", ["qwen3_8_flash_next", "qwen3_8_flash_next_text", "qwen4_exp"])
@pytest.mark.parametrize("text", ["CHECKING", "\n\n", ""])
@pytest.mark.parametrize("stored", [False, True])
def test_reasoning_commentary_and_calls_are_one_turn(model, text, stored):
    output = construct([reasoning(), message(text), call(), tool_result()], model, stored)
    assert [m["role"] for m in output] == ["user", "assistant", "tool"]
    assistant = output[1]
    assert assistant["reasoning_content"] == "PLAN"
    assert assistant.get("content", "") == text
    assert assistant["phase"] == "commentary"
    assert assistant["tool_calls"][0]["function"]["name"] == "inspect"
    assert output[2]["tool_call_id"] == "call_1"


@pytest.mark.parametrize("model", ["llama", "qwen3", "qwen2_5_vl"])
def test_other_models_keep_existing_grouping(model):
    output = construct([reasoning(), message(), call(), tool_result()], model)
    assert [m["role"] for m in output] == ["user", "assistant", "assistant", "assistant", "tool"]
    assert [m.get("phase") for m in output[1:4]] == [None, "commentary", None]


@pytest.mark.parametrize("items,expected", [
    ([reasoning(), message("REPORT", "final_answer")], [("PLAN", "REPORT", "final_answer", 0)]),
    ([reasoning(), call()], [("PLAN", "", None, 1)]),
    ([reasoning(), message(), reasoning("REPLAN"), call()],
     [("PLAN", "CHECKING", "commentary", 0), ("REPLAN", "", None, 1)]),
    ([reasoning(), message(), call(), message("REPORT", "final_answer")],
     [("PLAN", "CHECKING", "commentary", 1), ("", "REPORT", "final_answer", 0)]),
    ([call(), reasoning(), message("REPORT", "final_answer")],
     [("", "", None, 1), ("PLAN", "REPORT", "final_answer", 0)]),
    ([call(), message("AFTER", None)], [("", "", None, 1), ("", "AFTER", None, 0)]),
    ([message("REPORT", "final_answer"), call()],
     [("", "REPORT", "final_answer", 0), ("", "", None, 1)]),
    ([message("REPORT", "final_answer"), reasoning(), call()],
     [("", "REPORT", "final_answer", 0), ("PLAN", "", None, 1)]),
    ([message(), message("REPORT", "final_answer")],
     [("", "CHECKING", "commentary", 0), ("", "REPORT", "final_answer", 0)]),
    ([reasoning(), reasoning("MORE"), message(), call(), call("verify", "call_2")],
     [("PLAN\nMORE", "CHECKING", "commentary", 2)]),
    ([reasoning(), message(), call(), message("AFTER", "commentary"), call("verify", "call_2")],
     [("PLAN", "CHECKING", "commentary", 1), ("", "AFTER", "commentary", 1)]),
])
def test_order_restarts_and_explicit_phases_remain_boundaries(items, expected):
    output = construct(items)
    actual = [(m.get("reasoning_content", ""), m.get("content", ""), m.get("phase"),
               len(m.get("tool_calls", []))) for m in output if m["role"] == "assistant"]
    assert actual == expected


def test_namespaced_and_custom_calls_stay_with_reasoning_and_result_ids():
    custom = {"type": "custom_tool_call", "call_id": "call_2", "name": "patch", "input": "PATCH"}
    output = construct([reasoning(), message(), call(namespace="lab"), custom,
                        tool_result(), tool_result("call_2")])
    assert [m["role"] for m in output] == ["user", "assistant", "tool", "tool"]
    calls = output[1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["lab.inspect", "patch"]
    assert [c["id"] for c in calls] == ["call_1", "call_2"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"input": "PATCH"}


def test_media_and_encoded_reasoning_survive_grouping():
    from sglang.srt.entrypoints.openai.responses_adapters import encode_reasoning_state

    parts = [{"type": "output_text", "text": "CHECKING"},
             {"type": "input_image", "image_url": "https://example.invalid/image.png"}]
    output = construct([{"type": "reasoning", "encrypted_content": encode_reasoning_state("PLAN")},
                        message(parts), call(), tool_result()])
    assert len(output) == 3
    assert output[1]["reasoning_content"] == "PLAN"
    assert output[1]["content"][1]["image_url"]["url"] == "https://example.invalid/image.png"


@pytest.mark.parametrize("boundary", [
    {"role": "user", "content": "NEW QUERY"},
    {"role": "developer", "content": "NEW INSTRUCTION"},
    tool_result(),
])
def test_non_assistant_items_prevent_grouping(boundary):
    output = construct([reasoning(), message(), boundary, reasoning("SECOND"), message("REPORT", "final_answer")])
    assistants = [m for m in output if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert [m["reasoning_content"] for m in assistants] == ["PLAN", "SECOND"]


def test_native_combined_messages_keep_order_and_payloads():
    native = {"role": "assistant", "reasoning_content": "PLAN", "content": "CHECKING",
              "tool_calls": [{"id": "call_1", "type": "function", "function": {
                  "name": "inspect", "arguments": '{"value":"MARKER"}'}}]}
    output = construct([native, reasoning("SECOND"), message("REPORT", "final_answer")])
    assert output[1] == native
    assert output[2]["reasoning_content"] == "SECOND"
    assert output[2]["content"] == "REPORT"


def test_native_reasoning_and_content_accept_following_calls():
    native = {"role": "assistant", "reasoning_content": "PLAN", "content": "CHECKING"}
    serving = make_serving()
    messages = [native, serving._normalize_response_message_for_chat(call())]
    before = copy.deepcopy(messages)
    output = serving._merge_consecutive_assistant_messages(messages, preserve_qwen_order=True)
    assert messages == before
    assert len(output) == 1
    assert output[0]["reasoning_content"] == "PLAN"
    assert output[0]["content"] == "CHECKING"
    assert len(output[0]["tool_calls"]) == 1


def test_combined_native_reasoning_cannot_move_ahead_of_prior_content():
    native = {"role": "assistant", "reasoning_content": "SECOND PLAN", "content": "SECOND TEXT"}
    serving = make_serving()
    messages = [message("FIRST TEXT", None), native, serving._normalize_response_message_for_chat(call())]
    before = copy.deepcopy(messages)
    output = serving._merge_consecutive_assistant_messages(messages, preserve_qwen_order=True)
    assert messages == before
    assert len(output) == 2
    assert output[0]["content"] == "FIRST TEXT"
    assert output[1]["reasoning_content"] == "SECOND PLAN"
    assert output[1]["content"] == "SECOND TEXT"
    assert len(output[1]["tool_calls"]) == 1


def test_adjacent_list_and_string_content_preserve_parts():
    parts = [{"type": "output_text", "text": "FIRST"}]
    output = construct([reasoning(), message(parts), message("SECOND"), call()])
    assert len(output) == 2
    assert output[1]["content"] == [{"type": "text", "text": "FIRST"}, {"type": "text", "text": "SECOND"}]


@pytest.fixture(scope="module")
def pinned_tokenizer():
    path = os.environ.get("QWEN_REPLAY_MODEL_PATH")
    if not path:
        pytest.skip("Pinned Qwen tokenizer is provided by the local qualification runner")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("text", ["CHECKING", "\n\n", ""])
@pytest.mark.parametrize("stored", [False, True])
def test_pinned_template_matches_native_chat_turn(pinned_tokenizer, text, stored):
    actual = construct([reasoning(), message(text), call(), tool_result()], stored=stored)
    expected = [actual[0], {"role": "assistant", "reasoning_content": "PLAN", "content": text,
                 "tool_calls": [{"id": "call_1", "type": "function", "function": {
                     "name": "inspect", "arguments": {"value": "MARKER"}}}]}, actual[-1]]
    actual = copy.deepcopy(actual)
    for m in actual:
        normalize_assistant_tool_call_arguments(m)
    kwargs = dict(tokenize=False, add_generation_prompt=True, reasoning_effort="xhigh")
    rendered = pinned_tokenizer.apply_chat_template(actual, **kwargs)
    assert rendered == pinned_tokenizer.apply_chat_template(expected, **kwargs)
    assert rendered.count('<|im_start|>assistant') == 2  # History plus generation prefix.
    assert '</think>\n\n<|im_end|>' not in rendered


@pytest.mark.parametrize("items,markers", [
    ([reasoning("PLAN_FIRST"), message("TEXT_FIRST"), reasoning("PLAN_SECOND"), call()],
     ["PLAN_FIRST", "TEXT_FIRST", "PLAN_SECOND", "<function=inspect>"]),
    ([reasoning("PLAN_FIRST"), message("TEXT_FIRST"), call(), message("REPORT_LAST", "final_answer")],
     ["PLAN_FIRST", "TEXT_FIRST", "<function=inspect>", "REPORT_LAST"]),
    ([call(), reasoning("PLAN_SECOND"), message("REPORT_LAST", "final_answer")],
     ["<function=inspect>", "PLAN_SECOND", "REPORT_LAST"]),
    ([call(), message("TEXT_AFTER", "commentary"), call("verify", "call_2")],
     ["<function=inspect>", "TEXT_AFTER", "<function=verify>"]),
])
def test_pinned_template_retains_order_across_restarted_sequences(pinned_tokenizer, items, markers):
    actual = construct(items)
    for m in actual:
        normalize_assistant_tool_call_arguments(m)
    rendered = pinned_tokenizer.apply_chat_template(actual, tokenize=False, add_generation_prompt=True, reasoning_effort="xhigh")
    positions = [rendered.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert rendered.count('<|im_start|>assistant') == 3
