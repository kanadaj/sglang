"""Tool-result images must survive Responses validation and chat replay."""
import copy
import json

import pytest
from utils import make_serving
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ChatCompletionRequest

URL = "data:image/png;base64,iVBORw0KGgo="


def serving(model="qwen3_8_flash_next"):
    s = make_serving()
    s.tokenizer_manager.model_config.hf_config.model_type = model
    return s


def output(kind, detail=None):
    image = {"type": "input_image", "image_url": URL}
    if detail is not None:
        image["detail"] = detail
    return {"type": kind + "_call_output", "call_id": "call_1", "output": [
        {"type": "input_text", "text": "Before"}, image,
        {"type": "input_text", "text": "After"},
    ]}


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
@pytest.mark.parametrize("detail", [None, "low", "high"])
def test_validation_materializes_output_and_preserves_repeated_serialization(kind, detail):
    raw = output(kind, detail)
    before = copy.deepcopy(raw)
    req = ResponsesRequest(model="x", input=[raw])
    assert isinstance(req.input[0]["output"], list)
    first = req.model_dump_json()
    assert req.model_dump_json() == first
    serialized = json.loads(first)["input"][0]["output"]
    assert serialized[1]["image_url"] == URL
    assert serialized[1]["detail"] == (detail or "auto")
    assert raw == before


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("model", ["qwen3_8_flash_next", "qwen3_8_flash_next_text", "qwen4_exp"])
def test_tool_media_survives_stateless_and_stored_replay(kind, stored, model):
    s = serving(model)
    item = output(kind)
    req = ResponsesRequest(model="x", input=[item])
    # Logging/serialization before generation must not consume image content.
    req.model_dump_json()
    if stored:
        s.msg_store["resp_prior"] = req.input
        req = ResponsesRequest(model="x", input="Continue", previous_response_id="resp_prior")
    messages = s._construct_input_messages(req)
    tool = next(m for m in messages if m.get("role") == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert tool["content"] == [
        {"type": "text", "text": "Before"},
        {"type": "image_url", "image_url": {"url": URL, "detail": "auto"}},
        {"type": "text", "text": "After"},
    ]
    chat = ChatCompletionRequest(model="x", messages=messages)
    assert chat.messages[0].content[1].image_url.url == URL
    assert s._construct_input_messages(req) == messages


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
@pytest.mark.parametrize("value", ["plain", [], [{"type": "input_text", "text": "A"}, {"type": "input_text", "text": "B"}]])
def test_text_only_tool_output_keeps_existing_concatenation(kind, value):
    req = ResponsesRequest(model="x", input=[{"type": kind + "_call_output", "call_id": "c", "output": value}])
    messages = serving()._construct_input_messages(req)
    assert messages[0]["content"] == (value if isinstance(value, str) else "".join(p["text"] for p in value))


def test_non_qwen_retains_existing_conversion():
    req = ResponsesRequest(model="x", input=[output("function")])
    assert serving("llama")._construct_input_messages(req)[0]["content"] == "BeforeAfter"


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
def test_pinned_template_keeps_image_inside_tool_response(kind):
    import os
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_REPLAY_MODEL_PATH"], local_files_only=True)
    req = ResponsesRequest(model="x", input=[output(kind)])
    messages = serving()._construct_input_messages(req)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Inspect the tool image."}, *messages], tokenize=False, add_generation_prompt=True, reasoning_effort="xhigh")
    tool = rendered.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    assert tool.index("Before") < tool.index("<|image_pad|>") < tool.index("After")
    assert URL not in rendered
    assert rendered.count("<|image_pad|>") == 1


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
def test_unsupported_tool_media_is_rejected_instead_of_flattened(kind):
    from pydantic import ValidationError
    with pytest.raises((ValidationError, ValueError)):
        req = ResponsesRequest(model="x", input=[{"type": kind + "_call_output", "call_id": "c", "output": [{"type": "unsupported_media", "payload": "must not disappear"}]}])
        ChatCompletionRequest(model="x", messages=serving()._construct_input_messages(req))


def test_user_image_normalization_still_preserves_input():
    raw = {"role": "user", "content": [{"type": "input_image", "image_url": URL}]}
    before = copy.deepcopy(raw)
    req = ResponsesRequest(model="x", input=[raw])
    assert serving()._construct_input_messages(req)[0]["content"][0]["image_url"] == {"url": URL, "detail": "auto"}
    assert raw == before
