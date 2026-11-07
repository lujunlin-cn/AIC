import pytest

from aic.scale_runtime import format_scale_action, parse_scale_action, prompt_sha256, verify_tokenizer_roundtrip


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")

    def __len__(self):
        return 256


def test_scale_action_json_roundtrip_and_unique_tokenizer_actions():
    for scale_id in range(8):
        assert parse_scale_action(format_scale_action(scale_id), 8) == scale_id
    report = verify_tokenizer_roundtrip(TinyTokenizer(), 8)
    assert report["unique_action_token_sequences"] == 8
    assert report["new_tokens_added"] == 0
    assert prompt_sha256("same")==prompt_sha256("same")


@pytest.mark.parametrize("text", [
    "scale_id=2", '{"scale_id":true}', '{"scale_id":8}',
    '{"scale_id":1,"crop_center":[1,2]}', "{\"scale_id\":NaN}",
    '{"scale_id":1,"scale_id":2}',
])
def test_invalid_scale_action_output_is_rejected(text):
    with pytest.raises(ValueError):
        parse_scale_action(text, 8)
