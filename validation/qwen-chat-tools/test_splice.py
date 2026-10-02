"""Independent regression: deleting an unknown call cannot create a new call."""
import pytest
from test_schema import wire, tool, parsed


@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_unknown_wrapper_cannot_splice_a_declared_call(chunk):
    valid = wire()
    text = valid[:6] + wire('unknown') + valid[6:]
    assert parsed(text, [tool()], chunk) == []
