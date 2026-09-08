import torch

from aspectbench.privacy import checkpoint_tensor_inventory, membership_attack_report


def test_membership_attack_detects_separated_scores():
    report = membership_attack_report(
        [0.90, 0.85, 0.80, 0.75],
        [0.30, 0.25, 0.20, 0.10],
        bootstrap_samples=20,
        permutation_samples=20,
    )

    assert report["auc"] == 1.0
    assert report["maximum_tpr_minus_fpr"] == 1.0
    assert report["permutation_p_two_sided"] <= 1.0
    assert report["low_fpr_operating_points"]["1%"]["available"] is False


def test_membership_attack_reports_low_fpr_resolution():
    report = membership_attack_report(
        [0.9] * 100 + [0.1] * 100,
        list(torch.linspace(0.0, 0.5, 1000).numpy()),
        bootstrap_samples=10,
        permutation_samples=10,
    )
    at_one_percent = report["low_fpr_operating_points"]["1%"]
    assert at_one_percent["available"] is True
    assert at_one_percent["empirical_fpr"] <= 0.011
    assert at_one_percent["tpr"] >= 0.49
    assert report["low_fpr_operating_points"]["0.1%"]["available"] is True


def test_checkpoint_inventory_rejects_non_tensor_metadata(tmp_path):
    clean = tmp_path / "clean.pt"
    torch.save({"layer.weight": torch.ones(2, 3)}, clean)
    report = checkpoint_tensor_inventory(clean)

    assert report["tensor_only"] is True
    assert report["parameter_count"] == 6
    assert report["non_tensor_entries"] == {}
