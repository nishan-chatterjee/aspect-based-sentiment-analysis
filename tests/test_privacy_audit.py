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


def test_checkpoint_inventory_rejects_non_tensor_metadata(tmp_path):
    clean = tmp_path / "clean.pt"
    torch.save({"layer.weight": torch.ones(2, 3)}, clean)
    report = checkpoint_tensor_inventory(clean)

    assert report["tensor_only"] is True
    assert report["parameter_count"] == 6
    assert report["non_tensor_entries"] == {}
