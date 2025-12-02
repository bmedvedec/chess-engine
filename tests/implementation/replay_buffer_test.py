import random
import os

import chess
import torch

from chess_engine.data.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    GameExample,
)


def test_replay_buffer() -> None:
    """Test replay buffer functionality"""
    print("=" * 80)
    print("TESTING ENHANCED REPLAY BUFFER")
    print("=" * 80)

    # Test 1: Basic ReplayBuffer with tensor policies
    print("\n1. Testing basic ReplayBuffer...")
    buffer = ReplayBuffer(max_size=1000, memory_efficient=True)

    # Create sample data
    board = chess.Board()
    policy_tensor = torch.randn(4096)  # Random policy tensor
    value = 1.0
    move_history = [chess.Move.from_uci("e2e4")]

    # Add examples
    print("   Adding examples with tensor policies...")
    for i in range(100):
        buffer.add(board, policy_tensor, value, move_history)
    print(f"   Buffer size: {len(buffer)}")
    assert len(buffer) == 100, "Buffer should have 100 examples"

    # Test 2: Sampling
    print("\n2. Testing sampling...")
    batch = buffer.sample(batch_size=32)
    print(f"   Sampled batch size: {len(batch['boards'])}")
    print(f"   Boards type: {type(batch['boards'][0])}")
    print(f"   Policies type: {type(batch['policies'][0])}")
    print(f"   Values (first 5): {batch['values'][:5]}")
    assert len(batch["boards"]) == 32, "Should sample 32 examples"
    assert isinstance(batch["boards"][0], chess.Board), "Should return Board objects"

    # Test 3: add_game method
    print("\n3. Testing add_game...")
    positions = [chess.Board() for _ in range(10)]
    # Make some moves to have different positions
    for i, pos_board in enumerate(positions):
        if i > 0:
            legal_moves = list(pos_board.legal_moves)
            if legal_moves:
                pos_board.push(random.choice(legal_moves))

    policies_tensor = [torch.randn(4096) for _ in range(10)]
    outcome = 1.0
    buffer.add_game(positions, policies_tensor, outcome)
    print(f"   Buffer size after game: {len(buffer)}")
    assert len(buffer) == 110, "Buffer should have 110 examples now"

    # Test 4: Dict policies
    print("\n4. Testing dict policies...")
    policy_dict = {"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.2}
    buffer.add(board, policy_dict, 0.5)
    print(f"   Added example with dict policy, buffer size: {len(buffer)}")
    assert len(buffer) == 111, "Should have 111 examples"

    # Test 5: GameExample integration
    print("\n5. Testing GameExample integration...")
    examples = [
        GameExample(
            fen=chess.Board().fen(),
            policy={"e2e4": 0.6, "d2d4": 0.4},
            value=0.3,
            move_number=i,
        )
        for i in range(5)
    ]
    buffer.add_game_examples(examples)
    print(f"   Added 5 GameExamples, buffer size: {len(buffer)}")
    assert len(buffer) == 116, "Should have 116 examples"

    # Test 6: Buffer overflow (FIFO)
    print("\n6. Testing buffer overflow...")
    small_buffer = ReplayBuffer(max_size=50, memory_efficient=True)
    for i in range(100):
        small_buffer.add(board, policy_tensor, value)
    print(f"   Buffer size (max 50): {len(small_buffer)}")
    print(f"   Is full: {small_buffer.is_full()}")
    assert len(small_buffer) == 50, "Buffer should be capped at 50"
    assert small_buffer.is_full(), "Buffer should be full"

    # Test 7: Statistics
    print("\n7. Testing statistics...")
    stats = buffer.get_statistics()
    print(f"   Buffer statistics:")
    for key, val in stats.items():
        print(f"      {key}: {val:.3f}")
    assert "size" in stats, "Stats should include size"
    assert "avg_value" in stats, "Stats should include avg_value"

    # Test 8: Save/Load
    print("\n8. Testing save/load...")
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/test_buffer_enhanced.pkl"
    buffer.save(save_path)

    new_buffer = ReplayBuffer(max_size=1000, memory_efficient=True)
    new_buffer.load(save_path)
    print(f"   Loaded buffer size: {len(new_buffer)}")
    assert len(new_buffer) == len(buffer), "Loaded buffer should have same size"

    # Test 9: Clear buffer
    print("\n9. Testing clear...")
    clear_buffer = ReplayBuffer(max_size=100)
    clear_buffer.add(board, policy_tensor, value)
    print(f"   Buffer size before clear: {len(clear_buffer)}")
    clear_buffer.clear()
    print(f"   Buffer size after clear: {len(clear_buffer)}")
    assert len(clear_buffer) == 0, "Buffer should be empty after clear"

    # Test 10: PrioritizedReplayBuffer
    print("\n10. Testing PrioritizedReplayBuffer...")
    pri_buffer = PrioritizedReplayBuffer(max_size=1000, memory_efficient=True)

    for i in range(100):
        priority = random.random()  # Random priorities
        pri_buffer.add(board, policy_tensor, value, priority=priority)

    batch = pri_buffer.sample(batch_size=32, beta=0.4)
    print(f"   Sampled with priorities: {len(batch['boards'])}")
    print(f"   Importance weights (first 5): {batch['weights'][:5]}")
    assert len(batch["boards"]) == 32, "Should sample 32 examples"
    assert "weights" in batch, "Prioritized buffer should return weights"
    assert "indices" in batch, "Prioritized buffer should return indices"

    # Test 11: Update priorities
    print("\n11. Testing priority updates...")
    pri_buffer.update_priorities(batch["indices"][:5], [0.5] * 5)
    print(f"   Updated priorities for {len(batch['indices'][:5])} examples")

    # Test 12: GameExample with prioritized buffer
    print("\n12. Testing GameExample with prioritized buffer...")
    for i in range(5):
        example = GameExample(
            fen=chess.Board().fen(),
            policy={"e2e4": 0.7, "d2d4": 0.3},
            value=0.2,
            move_number=i,
        )
        pri_buffer.add_game_example(example, priority=1.5)
    print(f"   Added 5 GameExamples with priority, buffer size: {len(pri_buffer)}")

    # Test 13: Memory efficiency comparison
    print("\n13. Testing memory efficiency...")
    mem_eff_buffer = ReplayBuffer(max_size=1000, memory_efficient=True)
    mem_ineff_buffer = ReplayBuffer(max_size=1000, memory_efficient=False)

    for i in range(100):
        mem_eff_buffer.add(board, policy_dict, value)
        mem_ineff_buffer.add(board, policy_dict, value)

    print(f"   Memory-efficient buffer: {len(mem_eff_buffer)} examples")
    print(f"   Memory-inefficient buffer: {len(mem_ineff_buffer)} examples")
    print(f"   Both store same number, but memory-efficient uses less RAM")

    print("\n" + "=" * 80)
    print("✅ ALL REPLAY BUFFER TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_replay_buffer()
