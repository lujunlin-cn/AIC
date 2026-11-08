from scripts.check_scale_gate import evaluate


def test_scale_sft_gate_blocks_cuda_until_parent_and_live_contracts_are_verified():
    result = evaluate(
        "scale_sft",
        {"parent": {}, "training": {}, "data": {"scale_bins": {"values": [0.5, 1.0]}}},
        {"datasets": {}},
        {},
        {},
        {},
    )
    assert result["gate_passed"] is False
    assert result["cuda_initialization_allowed"] is False
    assert any("P1 base/adapter SHA256" in item for item in result["blockers"])
    assert any("LIVE source-level" in item for item in result["blockers"])


def test_candidate_gate_requires_sft_promotion_and_checkpoint_identity():
    result = evaluate("candidate_generation", {}, {}, {}, {}, {})
    assert result["gate_passed"] is False
    assert result["cuda_initialization_allowed"] is False
    assert any("SFT" in item for item in result["blockers"])
    assert any("smoke" in item for item in result["blockers"])


def test_scale_sft_gate_enforces_parameter_cap_and_dataset_use_review():
    config = {
        "parent": {
            "weights_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "parameter_count": 9_000_000_001,
        },
        "training": {
            "trainable_parameter_allowlist": ["spatial.scale_adapter"],
            "target_module_inventory_sha256": "d" * 64,
            "post_sft_inference_parameter_count": 8_900_000_000,
        },
        "data": {
            "scale_bins": {"count": 2, "values": [0.5, 1.0]},
            "minimum_position_conditioned_iou": 0.5,
            "training_sources": ["GNMC"],
        },
    }
    audit = {
        "datasets": {
            "LIVE_YT_VC": {"source_level_gate": True, "competition_use_authorized": True},
            "GNMC": {"source_split_disjoint": True, "competition_use_authorized": None},
            "GAICD": {"train_data_integrity_ready": True},
        },
    }
    result = evaluate("scale_sft", config, audit, {}, {}, {})
    assert result["cuda_initialization_allowed"] is False
    assert any("9B cap" in item for item in result["blockers"])
    assert any("GNMC competition-use license" in item for item in result["blockers"])


def test_downstream_gate_binds_actual_sft_and_candidate_checkpoint_identities():
    result = evaluate('spatial_dpo', {'training':{'sft_checkpoint_sha256':'actual'}}, {}, {},
                      {'checkpoint_sha256':'other'}, {'checkpoint_sha256':'third'})
    assert any('configured SFT/reference checkpoint' in item for item in result['blockers'])
    assert any('candidate and SFT metrics' in item for item in result['blockers'])
