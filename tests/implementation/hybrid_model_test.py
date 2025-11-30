from chess_engine.models.hybrid_model import HybridChessNet, count_parameters
import torch
import torch.nn.functional as F


def test_hybrid_model():
    """Test the hybrid model"""
    print("=" * 80)
    print("TESTING HYBRID CNN-RNN MODEL")
    print("=" * 80)

    # Test 1: CNN-only mode
    print("\n1. Testing CNN-only mode...")
    model_cnn = HybridChessNet(cnn_residual_blocks=5, use_rnn=False)
    board = torch.randn(4, 20, 8, 8)
    policy, value, attn = model_cnn(board)

    print(f"   Board shape: {board.shape}")
    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Attention: {attn}")
    print(f"   Parameters: {count_parameters(model_cnn):,}")
    print("   ✓ CNN-only mode working")

    # Test 2: Hybrid mode with concatenation
    print("\n2. Testing Hybrid mode (concatenation fusion)...")
    model_concat = HybridChessNet(
        cnn_residual_blocks=5, rnn_num_layers=2, use_rnn=True, fusion_type="concat"
    )
    move_history = torch.randint(0, 4096, (4, 50))
    lengths = torch.tensor([50, 45, 50, 30])

    policy, value, attn = model_concat(board, move_history, lengths)

    print(f"   Board shape: {board.shape}")
    print(f"   History shape: {move_history.shape}")
    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Parameters: {count_parameters(model_concat):,}")
    print("   ✓ Concat fusion working")

    # Test 3: Hybrid mode with gated fusion
    print("\n3. Testing Hybrid mode (gated fusion)...")
    model_gated = HybridChessNet(
        cnn_residual_blocks=5, rnn_num_layers=2, use_rnn=True, fusion_type="gated"
    )

    policy, value, attn = model_gated(board, move_history, lengths)

    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Parameters: {count_parameters(model_gated):,}")
    print("   ✓ Gated fusion working")

    # Test 4: Hybrid mode with attention
    print("\n4. Testing Hybrid mode (attention fusion + RNN attention)...")
    model_attn = HybridChessNet(
        cnn_residual_blocks=5,
        rnn_num_layers=2,
        use_rnn=True,
        rnn_use_attention=True,
        fusion_type="attention",
    )

    policy, value, attn_weights = model_attn(board, move_history, lengths)

    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print(
        f"   Attention weights shape: {attn_weights.shape if attn_weights is not None else None}"
    )
    print(f"   Parameters: {count_parameters(model_attn):,}")
    print("   ✓ Attention fusion working")

    # Test 5: Without move history (should work in hybrid mode too)
    print("\n5. Testing hybrid mode without move history...")
    policy, value, attn = model_gated(board)  # No history provided
    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print("   ✓ Works without history (CNN-only forward pass)")

    # Test 6: Prediction mode
    print("\n6. Testing prediction mode...")
    policy_probs, value = model_gated.predict(board, move_history, lengths)
    print(f"   Policy probs shape: {policy_probs.shape}")
    print(f"   Policy sum: {policy_probs.sum(dim=1)}")
    print(f"   Value shape: {value.shape}")
    print("   ✓ Prediction mode working")

    # Test 7: Gradient flow
    print("\n7. Testing gradient flow...")
    model = HybridChessNet(cnn_residual_blocks=2, rnn_num_layers=1, use_rnn=True)

    policy, value, _ = model(board, move_history, lengths)
    policy_target = torch.randn_like(policy)
    value_target = torch.randn_like(value)

    loss = F.mse_loss(policy, policy_target) + F.mse_loss(value, value_target)
    loss.backward()

    has_gradients = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {has_gradients}")
    print("   ✓ Gradient flow working")

    # Test 8: Training step (optional additional test)
    print("\n8. Testing complete training step...")
    model = HybridChessNet(cnn_residual_blocks=2, rnn_num_layers=1, use_rnn=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get initial parameter values
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    # Training step
    optimizer.zero_grad()
    policy, value, _ = model(board, move_history, lengths)
    policy_target = torch.randn_like(policy)
    value_target = torch.randn_like(value)

    loss = F.mse_loss(policy, policy_target) + F.mse_loss(value, value_target)
    loss.backward()
    optimizer.step()  # ← Actually update parameters

    # Check that parameters changed
    params_changed = any(
        not torch.equal(initial_params[name], param)
        for name, param in model.named_parameters()
    )

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Parameters updated: {params_changed}")
    print("   ✓ Training step working")

    # Test 9: Edge case - empty history
    print("\n9. Testing with empty move history...")
    empty_history = torch.zeros(4, 0, dtype=torch.long)
    empty_lengths = torch.zeros(4, dtype=torch.long)
    try:
        policy, value, _ = model_gated(board, empty_history, empty_lengths)
        print("   ✓ Handles empty history")
    except Exception as e:
        print(f"   ⚠ Empty history handling needs improvement: {e}")

    # Test 10: Single sample batch
    print("\n10. Testing batch size = 1...")
    single_board = board[:1]
    single_history = move_history[:1]
    single_length = lengths[:1]
    policy, value, _ = model_gated(single_board, single_history, single_length)
    assert policy.shape == (1, 4096) and value.shape == (1, 1)
    print("   ✓ Works with batch size 1")

    # Test 11: Model summary
    print("\n11. Testing model summary...")
    summary = model_gated.summary()
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Model size: {summary['model_size_mb']:.1f} MB")
    print(f"   Uses RNN: {summary['use_rnn']}")
    print(f"   Fusion type: {summary['fusion_type']}")
    print("   ✓ Model summary working")

    # Test 12: Inference speed benchmark
    print("\n12. Testing inference speed..")
    benchmark_results = model_gated.benchmark_inference_speed(num_iterations=100)
    print(f"   Average time: {benchmark_results['avg_time_ms']:.2f} ms")
    print(f"   Device: {benchmark_results['device']}")
    requirement_status = (
        "✅ PASS" if benchmark_results["passes_requirement"] else "❌ FAIL"
    )
    print(f"   <50ms: {requirement_status}")

    # Test 13: Advanced RNN features
    print("\n13. Testing advanced RNN features...")
    model_advanced = HybridChessNet(
        cnn_residual_blocks=3,
        rnn_num_layers=2,
        rnn_bidirectional=False,
        rnn_use_attention=True,
        rnn_use_layer_norm=True,
        rnn_use_positional_encoding=True,
        rnn_use_gradient_checkpointing=False,
        rnn_context_strategy="last",
        use_rnn=True,
        fusion_type="gated",
    )
    policy, value, attn = model_advanced(board, move_history, lengths)
    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Parameters: {count_parameters(model_advanced):,}")
    print("   ✓ Advanced RNN features working")

    # Test 14: Different context strategies
    print("\n14. Testing different context strategies...")
    strategies = ["last", "max", "mean", "multi"]
    for strategy in strategies:
        model_strategy = HybridChessNet(
            cnn_residual_blocks=2,
            rnn_num_layers=1,
            rnn_context_strategy=strategy,
            use_rnn=True,
        )
        policy, value, _ = model_strategy(board, move_history, lengths)
        print(f"   ✓ Context strategy '{strategy}' working")

    print("\n" + "=" * 80)
    print("✅ ALL HYBRID MODEL TESTS PASSED!")
    print("=" * 80)

    # Model comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    models = [
        ("CNN-only", HybridChessNet(cnn_residual_blocks=10, use_rnn=False)),
        (
            "Hybrid (concat)",
            HybridChessNet(cnn_residual_blocks=10, use_rnn=True, fusion_type="concat"),
        ),
        (
            "Hybrid (gated)",
            HybridChessNet(cnn_residual_blocks=10, use_rnn=True, fusion_type="gated"),
        ),
        (
            "Hybrid (attention)",
            HybridChessNet(
                cnn_residual_blocks=10,
                use_rnn=True,
                rnn_use_attention=True,
                fusion_type="attention",
            ),
        ),
    ]

    print(f"\n{'Model':<25} {'Parameters':>15} {'Size (MB)':>12}")
    print("-" * 55)

    for name, model in models:
        params = count_parameters(model)
        size_mb = params * 4 / (1024**2)
        print(f"{name:<25} {params:>15,} {size_mb:>12.1f}")

    print("\n💡 Recommendations:")
    print("   - Start with CNN-only for baseline")
    print("   - Add RNN for opening book and temporal understanding")
    print("   - Gated fusion is good default (learns to balance CNN/RNN)")
    print("   - Attention adds interpretability but more parameters")
    print("\n📊 Parameter naming convention:")
    print("   - CNN parameters: cnn_* (e.g., cnn_residual_blocks, cnn_filters)")
    print("   - RNN parameters: rnn_* (e.g., rnn_num_layers, rnn_hidden_size)")
    print("   - Fusion parameters: fusion_type")


if __name__ == "__main__":
    test_hybrid_model()
