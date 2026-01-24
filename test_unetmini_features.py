"""
Test script to verify CoordConv and Learnable Positional Encoding in UNetMini.
CoordConv is applied at TWO locations:
  1. At input (before enc1)
  2. At bottleneck (after enc3, before center decoder)
LPE is applied at the bottleneck (after enc3).
"""

import torch

from models.unetmini import UNetMini


def test_unetmini_with_features():
    """Test UNetMini with CoordConv and LPE options."""
    batch_size = 2
    input_channels = 3
    output_channels = 2
    input_size = (256, 256)

    print("Testing UNetMini with different configurations...")
    print("Note: CoordConv at INPUT + BOTTLENECK (2 places)")
    print("      LPE at BOTTLENECK\\n")

    # Test 1: Baseline (no CoordConv, no LPE)
    print("\n1. Baseline UNetMini (no CoordConv, no LPE)")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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
    print("\n2. UNetMini with Cartesian CoordConv")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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
    print("\n3. UNetMini with Radial CoordConv")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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
    print("\n4. UNetMini with LPE (add mode)")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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
    print("\n5. UNetMini with LPE (concat mode)")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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
    print("\n6. UNetMini with Cartesian CoordConv + LPE (add)")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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

    # Test 7: With Radial CoordConv and LPE (concat)
    print("\n7. UNetMini with Radial CoordConv + LPE (concat)")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
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

    # Test 8: Prototype mode with features
    print("\n8. UNetMini in prototype mode with features")
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size,
        prototype=True,
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

    # Test 9: Different input size
    print("\n9. UNetMini with different input size (128x128)")
    input_size_small = (128, 128)
    x_small = torch.randn(batch_size, input_channels, *input_size_small)
    model = UNetMini(
        input_channels=input_channels,
        output_channels=output_channels,
        input_size=input_size_small,
        coord_conv="cartesian",
        learnable_pe="concat",
    )
    out = model(x_small)
    print(f"   Input shape: {x_small.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (
        batch_size,
        output_channels,
        *input_size_small,
    ), "Output shape mismatch!"
    print("   ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_unetmini_with_features()
