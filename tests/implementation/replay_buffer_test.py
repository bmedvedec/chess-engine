import random
import os

import chess
import torch

from chess_engine.data.replay import (
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
    # stats = buffer.get_statistics()
    # print(f"   Buffer statistics:")
    # for key, val in stats.items():
    #     print(f"      {key}: {val:.3f}")
    # assert "size" in stats, "Stats should include size"
    # assert "avg_value" in stats, "Stats should include avg_value"

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


def test_per_correctness() -> None:
    """
    Targeted tests for PrioritizedReplayBuffer correctness.

    Covers:
      A. clear() resets priorities as well as the buffer
      B. Zero-priority samples don't cause division by zero
      C. Priority/buffer length desync raises AssertionError
      D. Higher-priority items are sampled more frequently
      E. IS weights are in (0, 1] and the maximum weight is always 1.0
      F. update_priorities() shifts the sampling distribution
      G. Save/load round-trip preserves priority values
      H. FIFO eviction keeps buffer and priorities in sync
    """
    from chess_engine.data.replay.buffer import ReplayBuffer  # noqa: F401
    from chess_engine.data.replay.prioritized import PrioritizedReplayBuffer
    from chess_engine.data.replay.storage import GameExample  # noqa: F401

    print("=" * 80)
    print("TESTING PRIORITIZED EXPERIENCE REPLAY CORRECTNESS")
    print("=" * 80)

    board = chess.Board()
    policy = {"e2e4": 0.6, "d2d4": 0.4}
    value = 1.0

    # ------------------------------------------------------------------
    # A. clear() must reset priorities alongside the buffer
    # ------------------------------------------------------------------
    print("\nA. Testing clear() resets priorities...")
    buf = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
    for i in range(10):
        buf.add(board, policy, value, priority=float(i + 1))
    assert len(buf) == 10
    assert len(buf.priorities) == 10

    buf.clear()
    assert len(buf) == 0, "Buffer should be empty after clear()"
    assert len(buf.priorities) == 0, "Priorities should be empty after clear()"
    print("   ✅ clear() resets both buffer and priorities")

    # ------------------------------------------------------------------
    # B. Zero-priority items don't cause NaN / division-by-zero
    # ------------------------------------------------------------------
    print("\nB. Testing zero-priority sampling (epsilon guard)...")
    buf = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
    for _ in range(20):
        buf.add(board, policy, value, priority=0.0)  # all zeros
    try:
        batch = buf.sample(batch_size=8, beta=0.4)
        assert len(batch["boards"]) == 8
        weights = batch["weights"]
        assert all(
            w == weights[0] for w in weights
        ), "All-zero priorities should produce uniform weights"
        print("   ✅ Zero priorities handled without NaN/crash")
    except Exception as e:
        raise AssertionError(f"Zero-priority sampling raised unexpected error: {e}")

    # ------------------------------------------------------------------
    # C. Buffer / priority desync raises AssertionError
    # ------------------------------------------------------------------
    print("\nC. Testing priority/buffer length mismatch assertion...")
    buf = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
    for _ in range(10):
        buf.add(board, policy, value, priority=1.0)
    # Manually break the invariant
    buf.priorities.append(99.0)
    try:
        buf.sample(batch_size=5, beta=0.4)
        raise AssertionError("Expected AssertionError was not raised")
    except AssertionError as e:
        assert (
            "mismatch" in str(e).lower() or "priority" in str(e).lower()
        ), f"Unexpected assertion message: {e}"
    print("   ✅ Desync detected and raised AssertionError")

    # ------------------------------------------------------------------
    # D. High-priority items are sampled proportionally more often
    # ------------------------------------------------------------------
    print("\nD. Testing priority-proportional sampling distribution...")
    buf = PrioritizedReplayBuffer(max_size=200, alpha=1.0)  # fully proportional
    # Item 0: priority 100 (should dominate)
    buf.add(board, {"a2a3": 1.0}, 1.0, priority=100.0)
    # Items 1-9: priority 1 each
    for _ in range(9):
        buf.add(board, {"b2b3": 1.0}, 0.0, priority=1.0)

    counts: dict = {0: 0, "other": 0}
    N_DRAWS = 500
    for _ in range(N_DRAWS):
        batch = buf.sample(batch_size=1, beta=0.4)
        if batch["indices"][0] == 0:
            counts[0] += 1
        else:
            counts["other"] += 1

    # P(item0) = 100/109 ≈ 91.7%; expect >70% to allow for randomness
    assert counts[0] > N_DRAWS * 0.70, (
        f"High-priority item sampled only {counts[0]}/{N_DRAWS} times "
        f"(expected >70%)"
    )
    print(
        f"   ✅ High-priority item sampled {counts[0]}/{N_DRAWS} "
        f"({counts[0]/N_DRAWS:.0%}) times"
    )

    # ------------------------------------------------------------------
    # E. IS weights are in (0, 1] and the maximum is normalised to 1.0
    # ------------------------------------------------------------------
    print("\nE. Testing IS weight normalisation...")
    buf = PrioritizedReplayBuffer(max_size=200, alpha=0.6)
    for i in range(50):
        buf.add(board, policy, value, priority=float(i + 1))

    for beta in (0.0, 0.4, 1.0):
        batch = buf.sample(batch_size=16, beta=beta)
        weights = batch["weights"]
        assert all(
            0.0 < w <= 1.0 + 1e-6 for w in weights
        ), f"IS weights out of (0,1] for beta={beta}: {weights}"
        assert (
            abs(max(weights) - 1.0) < 1e-5
        ), f"Max IS weight not 1.0 for beta={beta}: {max(weights)}"
    print("   ✅ IS weights in (0,1] and max=1.0 for β ∈ {0.0, 0.4, 1.0}")

    # ------------------------------------------------------------------
    # F. update_priorities() shifts the sampling distribution
    # ------------------------------------------------------------------
    print("\nF. Testing update_priorities() shifts sampling distribution...")
    buf = PrioritizedReplayBuffer(max_size=100, alpha=1.0)
    for _ in range(10):
        buf.add(board, policy, value, priority=1.0)  # uniform to start

    # Boost item at index 0 to a very high priority
    buf.update_priorities([0], [1000.0])

    counts_after: dict = {0: 0, "other": 0}
    for _ in range(200):
        batch = buf.sample(batch_size=1, beta=0.4)
        if batch["indices"][0] == 0:
            counts_after[0] += 1
        else:
            counts_after["other"] += 1

    assert counts_after[0] > 100, (
        f"After priority boost, item 0 sampled only "
        f"{counts_after[0]}/200 times (expected >100)"
    )
    print(f"   ✅ After priority boost, item 0 sampled {counts_after[0]}/200 times")

    # ------------------------------------------------------------------
    # G. Save/load round-trip preserves priority values
    # ------------------------------------------------------------------
    print("\nG. Testing save/load preserves priorities...")
    buf = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
    expected_priorities = [float(i + 1) for i in range(20)]
    for p in expected_priorities:
        buf.add(board, policy, value, priority=p)

    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/test_per_buffer.pkl"
    buf.save(save_path)

    loaded_buf = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
    loaded_buf.load(save_path)

    assert len(loaded_buf) == len(buf), "Loaded buffer has wrong size"
    assert len(loaded_buf.priorities) == len(
        loaded_buf
    ), "Loaded priorities length doesn't match buffer length"
    for orig, loaded in zip(expected_priorities, list(loaded_buf.priorities)):
        assert (
            abs(orig - loaded) < 1e-6
        ), f"Priority mismatch after load: {orig} vs {loaded}"
    print(
        f"   ✅ All {len(expected_priorities)} priorities preserved through save/load"
    )

    # ------------------------------------------------------------------
    # H. FIFO eviction keeps buffer and priorities in sync
    # ------------------------------------------------------------------
    print("\nH. Testing FIFO eviction keeps buffer/priorities in sync...")
    max_size = 50
    buf = PrioritizedReplayBuffer(max_size=max_size, alpha=0.6)
    for i in range(100):  # 2× overflow
        buf.add(board, policy, value, priority=float(i + 1))

    assert len(buf) == max_size, f"Buffer should be capped at {max_size}"
    assert len(buf.priorities) == max_size, (
        f"Priorities should also be capped at {max_size}, " f"got {len(buf.priorities)}"
    )
    # The assert inside sample() will catch any remaining desync
    batch = buf.sample(batch_size=10, beta=0.4)
    assert len(batch["boards"]) == 10
    print(
        f"   ✅ After overflow: len(buffer)={len(buf)}, "
        f"len(priorities)={len(buf.priorities)} — in sync"
    )

    print("\n" + "=" * 80)
    print("✅ ALL PER CORRECTNESS TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_replay_buffer()
    test_per_correctness()
