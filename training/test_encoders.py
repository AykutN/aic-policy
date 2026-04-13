# training/test_encoders.py
import torch

def test_image_encoder_output_shape():
    from training.models.encoders import ImageEncoder
    enc = ImageEncoder(feature_dim=512)
    # (B, To_img, n_cams, C, H, W)
    x = torch.zeros(2, 4, 3, 3, 256, 288)
    out = enc(x)
    assert out.shape == (2, 512), f"Got {out.shape}"


def test_ft_encoder_output_shape():
    from training.models.encoders import FTEncoder
    enc = FTEncoder(input_dim=6, window=16, feature_dim=128)
    x = torch.zeros(2, 16, 6)
    out = enc(x)
    assert out.shape == (2, 128), f"Got {out.shape}"


def test_proprio_encoder_output_shape():
    from training.models.encoders import ProprioEncoder
    enc = ProprioEncoder(input_dim=34, feature_dim=64)
    x = torch.zeros(2, 16, 34)
    out = enc(x)
    assert out.shape == (2, 64), f"Got {out.shape}"


if __name__ == "__main__":
    test_image_encoder_output_shape()
    test_ft_encoder_output_shape()
    test_proprio_encoder_output_shape()
    print("All encoder tests passed.")
