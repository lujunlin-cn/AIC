"""Strict scale-action serialization and tokenizer compatibility checks."""

from __future__ import annotations

import hashlib
import json


def format_scale_action(scale_id: int) -> str:
    if not isinstance(scale_id, int) or isinstance(scale_id, bool) or scale_id < 0:
        raise ValueError("scale_id must be a nonnegative integer")
    return json.dumps({"scale_id": scale_id}, separators=(",", ":"), sort_keys=True)


def parse_scale_action(text: str, scale_count: int) -> int:
    if (not isinstance(text, str) or not isinstance(scale_count, int)
            or isinstance(scale_count, bool) or scale_count <= 0):
        raise ValueError("invalid action text or scale count")
    text = text.strip()

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate action key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("scale action must be JSON") from exc
    if not isinstance(value, dict) or set(value) != {"scale_id"}:
        raise ValueError("scale action schema must contain only scale_id")
    scale_id = value["scale_id"]
    if not isinstance(scale_id, int) or isinstance(scale_id, bool) or not 0 <= scale_id < scale_count:
        raise ValueError("scale_id is outside the fitted bin range")
    return scale_id


def verify_tokenizer_roundtrip(tokenizer, scale_count: int) -> dict:
    actions = {}
    for scale_id in range(scale_count):
        text = format_scale_action(scale_id)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        parsed = parse_scale_action(decoded, scale_count)
        if parsed != scale_id:
            raise ValueError(f"tokenizer round-trip changed scale action {scale_id}")
        key = tuple(int(token_id) for token_id in token_ids)
        if key in actions and actions[key] != scale_id:
            raise ValueError("tokenizer maps multiple scale ids to identical token sequences")
        actions[key] = scale_id
    return {
        "scale_count": scale_count,
        "unique_action_token_sequences": len(actions),
        "tokenizer_vocab_size": len(tokenizer),
        "new_tokens_added": 0,
        "actions": {str(scale_id): format_scale_action(scale_id) for scale_id in range(scale_count)},
        "tokenization_sha256": hashlib.sha256(
            json.dumps(sorted((list(ids), scale_id) for ids, scale_id in actions.items()),
                       sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def prompt_sha256(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def apply_scale_to_p1_predictions(p1_record: dict, actions_by_frame: dict, width: int,
                                  height: int, target_ratio, bins) -> dict:
    """Final integration: preserve selected frame IDs and the P1 deployed centers."""
    import copy
    from aic.scale_geometry import checked_scale_action, deployed_center
    output = copy.deepcopy(p1_record)
    frames = [p["frame"] for p in p1_record["predictions"]]
    if set(frames) != set(actions_by_frame) or len(frames) != len(set(frames)):
        raise ValueError("Scale actions must cover exactly the immutable P1 selected frames")
    for item in output["predictions"]:
        action = actions_by_frame[item["frame"]]
        if type(action) is not int or not 0 <= action < len(bins):
            raise ValueError("invalid Scale ID")
        center = deployed_center(item["bboxes"], target_ratio)
        box = checked_scale_action(width, height, target_ratio, center, bins[action])
        if not box["valid"]:
            raise ValueError("position_shift_invalid")
        item["bboxes"] = [int(v) for v in box["aic_bbox_xyw"]]
    return output
