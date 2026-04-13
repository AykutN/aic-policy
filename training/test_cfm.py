# training/test_cfm.py
import torch

def test_cfm_forward_shape():
    from training.models.cfm import CFMPolicy
    model = CFMPolicy(
        image_feature_dim=512,
        ft_feature_dim=128,
        proprio_feature_dim=64,
        fusion_dim=256,
        flow_hidden_dim=512,
        flow_layers=6,
        action_dim=20,
        action_chunk=32,
    )
    B = 2
    images = torch.zeros(B, 4, 3, 3, 256, 288)
    proprio = torch.zeros(B, 16, 34)
    ft = torch.zeros(B, 16, 6)
    actions_gt = torch.rand(B, 32, 20)

    loss = model.loss(images, proprio, ft, actions_gt)
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    # Inference
    pred = model.sample(images, proprio, ft, n_steps=5)
    assert pred.shape == (B, 32, 20), f"Got {pred.shape}"


if __name__ == "__main__":
    test_cfm_forward_shape()
    print("All CFM tests passed.")
