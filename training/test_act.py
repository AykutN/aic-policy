# training/test_act.py
import torch


def test_act_forward_shape():
    from training.models.act import ACTPolicy
    model = ACTPolicy(
        image_feature_dim=512,
        ft_feature_dim=128,
        proprio_feature_dim=64,
        transformer_dim=256,
        transformer_heads=4,
        transformer_layers=4,
        latent_dim=64,
        action_dim=20,
        action_chunk=32,
    )
    B = 2
    images = torch.zeros(B, 4, 3, 3, 256, 288)
    proprio = torch.zeros(B, 16, 34)
    ft = torch.zeros(B, 16, 6)
    actions_gt = torch.zeros(B, 32, 20)  # only used during training

    # Training forward (with CVAE encoder)
    pred_actions, mu, log_var = model(images, proprio, ft, actions_gt)
    assert pred_actions.shape == (B, 32, 20), f"Got {pred_actions.shape}"
    assert mu.shape == (B, 64), f"Got {mu.shape}"

    # Inference forward (no actions_gt → sample from prior)
    pred_actions_inf, _, _ = model(images, proprio, ft, actions_gt=None)
    assert pred_actions_inf.shape == (B, 32, 20)


def test_act_loss():
    from training.models.act import act_loss
    pred = torch.zeros(2, 32, 20)
    target = torch.ones(2, 32, 20)
    mu = torch.zeros(2, 64)
    log_var = torch.zeros(2, 64)
    loss = act_loss(pred, target, mu, log_var, kl_weight=0.1)
    assert loss.item() > 0


if __name__ == "__main__":
    test_act_forward_shape()
    test_act_loss()
    print("All ACT tests passed.")
