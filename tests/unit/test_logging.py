from research_agent.logging import redact_secrets


def test_redacts_sensitive_keys_case_insensitively() -> None:
    event = {
        "event": "x",
        "Authorization": "Bearer abc",
        "x-api-key": "k",
        "anthropic_api_key": "sk-ant-123",
        "headers": {"authorization": "Bearer zzz", "accept": "json"},
    }
    out = redact_secrets(None, "info", event)
    assert out["Authorization"] == "***"
    assert out["x-api-key"] == "***"
    assert out["anthropic_api_key"] == "***"
    assert out["headers"] == {"authorization": "***", "accept": "json"}


def test_redacts_key_like_values_inside_strings() -> None:
    out = redact_secrets(None, "info", {"event": "failed with key sk-ant-api03-AbCdEf123456"})
    assert "AbCdEf123456" not in out["event"]
    out = redact_secrets(None, "info", {"msg": "tvly-dev-ABCDEFGH12345678 leaked"})
    assert "ABCDEFGH12345678" not in out["msg"]


def test_leaves_normal_fields_untouched() -> None:
    event = {"event": "run finished", "tokens": 10, "run_id": "r1"}
    assert redact_secrets(None, "info", dict(event)) == event
