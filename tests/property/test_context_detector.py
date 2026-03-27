"""Property-based tests for ContextWindowDetector."""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.context_detector import ContextWindowDetector


class _FakeModel:
    def __init__(self, context_window):
        self.metadata = {"context_window": context_window}


# --- Property 7: Context Window 70% 可用比例 ---
# Feature: sub-agent-review, Property 7: Context Window 70% 可用比例

@given(context_window=st.integers(min_value=1, max_value=10_000_000))
@settings(max_examples=100)
def test_property_7_usable_ratio(context_window):
    """When model metadata provides context_window, result = int(value * 0.7).

    Validates: Requirements 5.3
    """
    model = _FakeModel(context_window)
    result = ContextWindowDetector.detect(model)
    expected = int(context_window * 0.7)
    assert result == expected
