"""Request-local effort compatibility for this dedicated Qwen3.8FN image.

No model weights, global configuration or shared default dictionaries are mutated.
All three API adapters converge on ChatCompletionRequest before template rendering.
"""

ALIASES = {"minimal": "low", "high": "xhigh", "max": "xhigh"}
LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def normalize_qwen_effort(request, defaults):
    defaults = defaults or {}
    explicit = dict(request.chat_template_kwargs or {})
    effort = explicit.get("reasoning_effort")
    if effort is None:
        effort = request.reasoning_effort
    if effort is None:
        effort = defaults.get("reasoning_effort", "medium")
    if not isinstance(effort, str) or effort not in LEVELS:
        raise ValueError(f"Unsupported Qwen reasoning effort: {effort!r}")
    if effort == "none" and explicit.get("enable_thinking") is True:
        raise ValueError("reasoning effort 'none' conflicts with enable_thinking=true")
    merged = dict(defaults)
    merged.update(explicit)
    effort = ALIASES.get(effort, effort)
    # Explicit disable (including the Anthropic adapter's native thinking toggle)
    # always stays off, even if the client also supplies an output effort level.
    if effort == "none":
        merged["enable_thinking"] = False
    merged["reasoning_effort"] = effort
    request.reasoning_effort = effort
    request.chat_template_kwargs = merged
