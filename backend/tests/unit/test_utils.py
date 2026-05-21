"""Unit tests for utility functions."""
import pytest
from unittest.mock import patch
from datetime import datetime, timezone


@pytest.mark.unit
class TestHelpers:
    def test_generate_id_returns_string(self):
        from app.utils.helpers import generate_id
        id1 = generate_id()
        id2 = generate_id()
        assert isinstance(id1, str)
        assert id1 != id2

    def test_now_utc_returns_timezone_aware(self):
        from app.utils.helpers import now_utc
        dt = now_utc()
        assert dt.tzinfo is not None

    def test_truncate_text_within_limit(self):
        from app.utils.helpers import truncate_text
        text = "Hello World"
        result = truncate_text(text, 100)
        assert result == text

    def test_truncate_text_over_limit(self):
        from app.utils.helpers import truncate_text
        text = "A" * 200
        result = truncate_text(text, 50)
        assert len(result) <= 53  # 50 + "..."

    def test_safe_json_loads_valid(self):
        from app.utils.helpers import safe_json_loads
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_safe_json_loads_invalid(self):
        from app.utils.helpers import safe_json_loads
        result = safe_json_loads("not json")
        assert result is None

    def test_chunk_list(self):
        from app.utils.helpers import chunk_list
        items = list(range(10))
        chunks = chunk_list(items, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]


@pytest.mark.unit
class TestValidators:
    def test_validate_email_valid(self):
        from app.utils.validators import validate_email
        assert validate_email("user@example.com") is True

    def test_validate_email_invalid(self):
        from app.utils.validators import validate_email
        assert validate_email("not-an-email") is False

    def test_sanitize_text_removes_scripts(self):
        from app.utils.validators import sanitize_text
        dirty = "<script>alert('xss')</script>Hello"
        clean = sanitize_text(dirty)
        assert "<script>" not in clean
        assert "Hello" in clean
