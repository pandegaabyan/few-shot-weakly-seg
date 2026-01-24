"""
Test script to verify CoordConv and Learnable Positional Encoding in DeepLabv3.
CoordConv is applied at THREE locations for DeepLabV3+:
  1. At input (before backbone)
  2. Before ASPP (high-level features in head)
  3. Before 1x1 conv/project (low-level features in decoder)
CoordConv is applied at TWO locations for DeepLabV3:
  1. At input (before backbone)
  2. Before ASPP (in head)
LPE is applied at TWO locations for DeepLabV3+:
  1. To low-level features (before 1x1 conv/project in decoder)
  2. To high-level features (before ASPP in head)
LPE is applied at ONE location for DeepLabV3:
  1. Before ASPP (in head)
"""

import torch

from models.deeplabv3.models import load_deeplabv3


def test_deeplabv3_with_features():
    """Test DeepLabV3 with CoordConv and LPE options."""
    batch_size = 2
    input_channels = 3
    output_channels = 2
    input_size = (256, 256)

    print("Testing DeepLabV3+ with different configurations...")
    print("Note: CoordConv at INPUT + HEAD (3 places for V3+, 2 for V3)")
    print("      LPE at LOW-LEVEL + HIGH-LEVEL (2 places for V3+, 1 for V3)\\n")

    # Test 1: Baseline (no CoordConv, no LPE)
    print("\n1. Baseline DeepLabV3+ (no CoordConv, no LPE)")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
    )
    x = torch.randn(batch_size, input_channels, *input_size)
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 2: With Cartesian CoordConv
    print("\n2. DeepLabV3+ with Cartesian CoordConv")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        coord_conv="cartesian",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 3: With Radial CoordConv
    print("\n3. DeepLabV3+ with Radial CoordConv")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        coord_conv="radial",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 4: With LPE (add mode)
    print("\n4. DeepLabV3+ with LPE (add mode)")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        learnable_pe="add",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 5: With LPE (concat mode)
    print("\n5. DeepLabV3+ with LPE (concat mode)")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        learnable_pe="concat",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 6: With both CoordConv and LPE
    print("\n6. DeepLabV3+ with CoordConv + LPE")
    model = load_deeplabv3(
        plus=True,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        coord_conv="cartesian",
        learnable_pe="add",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    # Test 7: Regular DeepLabV3 (not plus) with features
    print("\n7. DeepLabV3 (not plus) with CoordConv + LPE")
    model = load_deeplabv3(
        plus=False,
        backbone="resnet50",
        output_channels=output_channels,
        input_size=input_size,
        in_channels=input_channels,
        coord_conv="radial",
        learnable_pe="concat",
    )
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_deeplabv3_with_features()
