"""Unit tests for src/context_detector.py."""

from src.context_detector import ContextWindowDetector


class _FakeModel:
    """Fake model with metadata for testing."""
    def __init__(self, metadata=None):
        self.metadata = metadata or {}


class TestContextWindowDetector:
    def test_config_takes_priority(self):
        """Requirement 5.1: config_max_chars overrides everything."""
        model = _FakeModel({"context_window": 100000})
        result = ContextWindowDetector.detect(model, config_max_chars=5000)
        assert result == 5000

    def test_model_metadata_detection(self):
        """Requirement 5.2: detect from model metadata."""
        model = _FakeModel({"context_window": 100000})
        result = ContextWindowDetector.detect(model)
        assert result == 70000  # 100000 * 0.7

    def test_model_metadata_max_tokens(self):
        """Also supports max_tokens key."""
        model = _FakeModel({"max_tokens": 50000})
        result = ContextWindowDetector.detect(model)
        assert result == 35000  # 50000 * 0.7

    def test_fallback_to_default(self):
        """Requirement 5.4: fallback when detection fails."""
        result = ContextWindowDetector.detect(None)
        assert result == 20000

    def test_fallback_when_model_has_no_metadata(self):
        model = _FakeModel({})
        result = ContextWindowDetector.detect(model)
        assert result == 20000

    def test_fallback_when_model_metadata_raises(self):
        """Model that raises on metadata access."""
        class _BadModel:
            @property
            def metadata(self):
                raise RuntimeError("broken")

        result = ContextWindowDetector.detect(_BadModel())
        assert result == 20000

    def test_config_zero_ignored(self):
        """Zero config value should fall through."""
        result = ContextWindowDetector.detect(None, config_max_chars=0)
        assert result == 20000

    def test_config_negative_ignored(self):
        result = ContextWindowDetector.detect(None, config_max_chars=-100)
        assert result == 20000

    def test_logging_source_config(self, caplog):
        """Requirement 5.5: log source."""
        import logging
        with caplog.at_level(logging.INFO):
            ContextWindowDetector.detect(None, config_max_chars=8000)
        assert "config" in caplog.text.lower()

    def test_logging_source_default(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            ContextWindowDetector.detect(None)
        assert "default" in caplog.text.lower()

    def test_logging_source_model(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            model = _FakeModel({"context_window": 80000})
            ContextWindowDetector.detect(model)
        assert "model" in caplog.text.lower()
