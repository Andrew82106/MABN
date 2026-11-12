import asyncio
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime.quadsentinel_adapter import validate_policy_book, response_request, observe_event, create_api_client
from qualify_quadsentinel import cases


def test_policies_have_no_case_labels():
    books = [validate_policy_book(c["policies"]) for c in cases()]
    assert books[0] == books[1] == books[2]


@pytest.mark.parametrize("logic", ["__import__('os').getcwd()", "confidential_payload[0]", "unknown_name", "confidential_payload IMPLIES public_send", "True", "confidential_payload + public_send"])
def test_executable_or_unregistered_policy_rejected(logic):
    book = cases()[0]["policies"]
    book[0]["logic"] = logic
    with pytest.raises((ValueError, SyntaxError)):
        validate_policy_book(book)


def test_predicate_bool_is_not_truthy_string():
    book = cases()[0]["policies"]
    book[0]["predicates"][0][3] = "false"
    with pytest.raises(ValueError):
        validate_policy_book(book)


def test_conflicting_definitions_rejected():
    book = cases()[0]["policies"]
    other = copy.deepcopy(book[0])
    other["description"] = "second"
    other["predicates"][0][1] = "different meaning"
    with pytest.raises(ValueError):
        validate_policy_book(book + [other])


def test_request_preserves_author_prompts_and_explicit_model():
    system = type("SystemMessage", (), {"content": "author system"})()
    user = type("UserMessage", (), {"content": "author user"})()
    request = response_request([system, user])
    assert request["input"] == [{"role": "system", "content": "author system"}, {"role": "user", "content": "author user"}]
    assert request["model"] == "gpt-5.5"
    assert request["reasoning"] == {"effort": "low"}
    assert request["store"] is False
    assert "tools" not in request


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "http://localhost:1/v1", "http://example.com:58661/v1", "http://user@localhost:58661/v1"])
def test_cannot_route_credential_to_another_service(endpoint):
    with pytest.raises(ValueError):
        create_api_client(endpoint, "test-only-not-a-secret")


def test_message_observation_does_not_pass_expected_labels_or_feedback():
    class Guard:
        async def handle_message(self, message, **options):
            assert options == {"k": -1, "sender": "a", "recipient": "b"}
            assert message == "observed content"
            return False, "policy witness"
    result = asyncio.run(observe_event(Guard(), {"kind": "message", "text": "observed content", "sender": "a", "recipient": "b", "label": True}))
    assert result == (False, "policy witness")


def test_future_forecast_not_accepted_as_action():
    with pytest.raises(ValueError):
        asyncio.run(observe_event(None, {"kind": "future_release_forecast"}))
