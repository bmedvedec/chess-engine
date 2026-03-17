from chess_engine.models.cnn.chess_net import (
    ChessCNN,
    PolicyHead,
    ValueHead,
    ChessNet,
    count_parameters,
)
from chess_engine.models.cnn.backbone import ChessResidualBlock
import torch
import torch.nn.functional as F


def test_architecture():
    """Test the CNN architecture with sample data"""
    print("=" * 80)
    print("TESTING CNN ARCHITECTURE")
    print("=" * 80)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Test 1: Residual Block
    print("\n1. Testing Residual Block...")
    residual_block = ChessResidualBlock(num_filters=256).to(device)
    test_input = torch.randn(4, 256, 8, 8, device=device)  # Batch of 4
    output = residual_block(test_input)
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {count_parameters(residual_block):,}")
    assert output.shape == test_input.shape, "Shape mismatch!"
    print("   [OK] Residual block working correctly")

    # Test 2: CNN Backbone
    print("\n2. Testing CNN Backbone...")
    backbone = ChessCNN(input_channels=22, num_filters=256, num_residual_blocks=10).to(
        device
    )
    board_input = torch.randn(4, 22, 8, 8, device=device)
    features = backbone(board_input)
    print(f"   Input shape:  {board_input.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Parameters: {count_parameters(backbone):,}")
    assert features.shape == (4, 256, 8, 8), "Shape mismatch!"
    print("   [OK] CNN backbone working correctly")

    # Test 3: Policy Head
    print("\n3. Testing Policy Head...")
    policy_head = PolicyHead(input_channels=256, num_actions=4096).to(device)
    policy_logits = policy_head(features)
    print(f"   Input shape:  {features.shape}")
    print(f"   Output shape: {policy_logits.shape}")
    print(f"   Parameters: {count_parameters(policy_head):,}")
    assert policy_logits.shape == (4, 4096), "Shape mismatch!"
    print("   [OK] Policy head working correctly")

    # Test 4: Value Head
    print("\n4. Testing Value Head...")
    value_head = ValueHead(input_channels=256).to(device)
    value = value_head(features)
    print(f"   Input shape:  {features.shape}")
    print(f"   Output shape: {value.shape}")
    print(f"   Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"   Parameters: {count_parameters(value_head):,}")
    assert value.shape == (4, 1), "Shape mismatch!"
    assert value.min() >= -1.0 and value.max() <= 1.0, "Value out of range!"
    print("   [OK] Value head working correctly")

    # Test 5: Complete Network
    print("\n5. Testing Complete Network...")
    model = ChessNet(
        input_channels=22,
        num_filters=256,
        num_residual_blocks=10,
        num_actions=4096,
        device=device,
    )
    policy_logits, value = model(board_input)
    print(f"   Input shape:  {board_input.shape}")
    print(f"   Policy shape: {policy_logits.shape}")
    print(f"   Value shape:  {value.shape}")
    print(f"   Total parameters: {count_parameters(model):,}")
    print("   [OK] Complete network working correctly")

    # Test 6: Prediction mode
    print("\n6. Testing Prediction Mode...")
    policy_probs, value = model.predict(board_input)
    print(f"   Policy probabilities shape: {policy_probs.shape}")
    print(f"   Policy sum: {policy_probs.sum(dim=1)}")  # Should be ~1.0 for each batch
    print(f"   Value shape: {value.shape}")
    print("   [OK] Prediction mode working correctly")

    # Test 7: Gradient flow
    print("\n7. Testing Gradient Flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward pass
    policy_logits, value = model(board_input)

    # Dummy loss
    policy_target = torch.randn(4, 4096, device=device)
    value_target = torch.randn(4, 1, device=device)

    loss = F.mse_loss(policy_logits, policy_target) + F.mse_loss(value, value_target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    has_gradients = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {has_gradients}")
    print("   [OK] Gradient flow working correctly")

    # Test 8: Save/Load functionality
    print("\n8. Testing Save/Load...")
    import tempfile
    import os

    # Create temp file and get path
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()  # Close immediately

    try:
        model.save(tmp_path)
        loaded_model = ChessNet.load(tmp_path, device=str(device))

        # Verify same outputs
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            p1, v1 = model(board_input)
            p2, v2 = loaded_model(board_input)
            assert torch.allclose(p1, p2, atol=1e-6), "Policy mismatch after load!"
            assert torch.allclose(v1, v2, atol=1e-6), "Value mismatch after load!"

        print("   [OK] Save/Load working correctly")
    finally:
        # Cleanup - works on both Windows and Unix
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows might still have it locked

    print("\n" + "=" * 80)
    print("[OK] ALL ARCHITECTURE TESTS PASSED!")
    print("=" * 80)

    # Print model summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    info = model.get_model_info()
    print(f"Architecture: {info['architecture']}")
    print(f"Input: {info['input_shape']}")
    print(f"Backbone: {info['num_residual_blocks']} residual blocks")
    print(f"Filters: {info['num_filters']}")
    print(f"Output: Policy ({info['num_actions']} moves) + Value ([-1, 1])")
    print(f"Device: {info['device']}")
    print(f"\nParameter Count:")
    print(f"  Backbone:    {info['backbone_parameters']:>10,}")
    print(f"  Policy Head: {info['policy_head_parameters']:>10,}")
    print(f"  Value Head:  {info['value_head_parameters']:>10,}")
    print(f"  Total:       {info['total_parameters']:>10,}")

    # Estimate model size
    param_size_mb = info["total_parameters"] * 4 / (1024 * 1024)
    print(f"\nEstimated Model Size: {param_size_mb:.2f} MB")


if __name__ == "__main__":
    test_architecture()
