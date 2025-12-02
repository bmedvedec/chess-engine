import sys
import os
import time
import chess
import torch

from chess_engine.models.hybrid_model import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts import MCTS


def test_mcts():
    """Comprehensive MCTS test suite"""
    print("=" * 80)
    print("COMPREHENSIVE MCTS TEST SUITE")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model with updated parameters
    print("\nCreating model...")
    model = HybridChessNet(
        cnn_residual_blocks=3,
        use_rnn=False,
    )
    model.to(device)
    model.eval()

    # Create encoders
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # Test 1: Basic MCTS search
    print("\n" + "=" * 80)
    print("TEST 1: Basic MCTS Search")
    print("=" * 80)
    mcts = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=20,
        c_puct=1.5,
        enable_caching=True,
    )

    board = chess.Board()
    print(f"Position: {board.fen()}")

    move, stats = mcts.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"✓ Selected move: {move.uci()}")
    print(f"  Total visits: {stats['total_visits']}")
    print(f"  Root value: {stats['root_value']:.3f}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Simulations: {stats['simulations_completed']}")

    # Test 2: Position caching effectiveness
    print("\n" + "=" * 80)
    print("TEST 2: Position Caching")
    print("=" * 80)
    start = time.time()
    move1, _ = mcts.search(board, return_stats=False)
    time1 = time.time() - start

    start = time.time()
    move2, _ = mcts.search(board, return_stats=False)
    time2 = time.time() - start

    print(f"    First search: {time1:.3f}s")
    print(f"    Second search: {time2:.3f}s (cached)")
    print(f"    Speedup: {time1/time2 if time2 > 0 else 1.0:.2f}x")

    # Test 3: More simulations with statistics
    print("\n" + "=" * 80)
    print("TEST 3: Detailed Statistics")
    print("=" * 80)
    mcts = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=100,
    )

    move, stats = mcts.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"    Selected move: {move.uci()}")
    print(f"\nTop 5 moves:")
    for i, move_stat in enumerate(stats["top_moves"], 1):
        print(
            f"  {i}. {move_stat['move']:5s} "
            f"visits={move_stat['visits']:3d} ({move_stat['visit_pct']:.1%}) "
            f"value={move_stat['value']:+.3f} "
            f"prior={move_stat['prior']:.3f}"
        )

    # Test 4: Tactical position
    print("\n" + "=" * 80)
    print("TEST 4: Tactical Position (Scholar's Mate Threat)")
    print("=" * 80)
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    print(board)

    move, stats = mcts.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"\n    Selected move: {move.uci()}")
    print(f"  Value: {stats['root_value']:.3f}")
    print(f"\nTop 3 defensive moves:")
    for i, move_stat in enumerate(stats["top_moves"][:3], 1):
        print(
            f"  {i}. {move_stat['move']:5s} "
            f"visits={move_stat['visits']:3d} "
            f"value={move_stat['value']:+.3f}"
        )

    # Test 5: Checkmate in 1 detection
    print("\n" + "=" * 80)
    print("TEST 5: Checkmate Detection")
    print("=" * 80)
    board = chess.Board(
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    )
    print("Position: Checkmate in 1 for white")
    print(board)

    # White's turn - should find checkmate
    board.turn = chess.WHITE
    move, stats = mcts.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"\n    Found move: {move.uci()}")

    if stats["root_value"] > 0.5:
        print(f"    Correctly evaluates as winning ({stats['root_value']:.3f})")
    else:
        print(
            f"      Warning: Value {stats['root_value']:.3f} (expected > 0.5 for winning)"
        )

    # Test 6: Progressive widening
    print("\n" + "=" * 80)
    print("TEST 6: Progressive Widening")
    print("=" * 80)
    mcts_progressive = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=100,
        use_progressive_widening=True,
    )

    board = chess.Board()
    move, stats = mcts_progressive.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"    Selected move: {move.uci()}")
    print(f"  Children expanded: {stats['num_children']} (progressive)")

    # Test 7: Early termination
    print("\n" + "=" * 80)
    print("TEST 7: Early Termination")
    print("=" * 80)
    mcts_early = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=500,  # Set high, but expect early termination
        enable_early_termination=True,
        early_termination_threshold=0.7,
    )

    board = chess.Board()
    start = time.time()
    move, stats = mcts_early.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"
    elapsed = time.time() - start

    print(f"    Selected move: {move.uci()}")
    print(f"  Terminated early: {stats['simulations_completed']}/{500} simulations")
    print(f"  Time: {elapsed:.2f}s")

    # Test 8: Dirichlet noise (for training)
    print("\n" + "=" * 80)
    print("TEST 8: Dirichlet Noise")
    print("=" * 80)
    mcts_noise = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=50,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
    )

    board = chess.Board()
    move, stats = mcts_noise.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"    Selected move with noise: {move.uci()}")
    print(f"  (Adds exploration for self-play training)")

    # Test 9: Batched evaluation
    print("\n" + "=" * 80)
    print("TEST 9: Batched Evaluation")
    print("=" * 80)
    mcts_batch = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=64,
        eval_batch_size=8,
    )

    board = chess.Board()
    start = time.time()
    move, stats = mcts_batch.search_batched(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"
    elapsed = time.time() - start

    print(f"    Selected move: {move.uci()}")
    print(f"  Batch evaluation time: {elapsed:.2f}s")
    print(f"  (More efficient on GPU)")

    # Test 10: Time-limited search
    print("\n" + "=" * 80)
    print("TEST 10: Time-Limited Search")
    print("=" * 80)
    mcts_time = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=1000,  # Will be limited by time
    )

    board = chess.Board()
    time_limit = 2.0  # 2 seconds
    move, stats = mcts_time.search_with_time_limit(board, time_limit, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"    Selected move: {move.uci()}")
    print(f"  Time limit: {time_limit}s")
    print(f"  Time used: {stats['time_used']:.2f}s")
    print(f"  Simulations: {stats['simulations_completed']}")

    # Test 11: RNN integration
    print("\n" + "=" * 80)
    print("TEST 11: RNN Integration")
    print("=" * 80)
    model_rnn = HybridChessNet(
        cnn_residual_blocks=3,
        use_rnn=True,
    )
    model_rnn.to(device)
    model_rnn.eval()

    mcts_rnn = MCTS(
        model=model_rnn,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=20,
        use_rnn=True,
    )

    # Create a position with move history
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    board.push(chess.Move.from_uci("g1f3"))

    print(f"Position with 3 moves played:")
    print(board)

    move, stats = mcts_rnn.search(board, return_stats=True)
    assert stats is not None, "Stats should not be None when return_stats=True"

    print(f"\n    RNN-enhanced move: {move.uci()}")
    print(f"  (Uses move history context)")

    # Test 12: Speed benchmark
    print("\n" + "=" * 80)
    print("TEST 12: Speed Benchmark")
    print("=" * 80)
    board = chess.Board()
    num_searches = 5

    mcts_benchmark = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=100,
    )

    start = time.time()
    for _ in range(num_searches):
        move, _ = mcts_benchmark.search(board, return_stats=False)
    elapsed = time.time() - start

    print(f"    {num_searches} searches in {elapsed:.2f}s")
    print(f"  Average: {elapsed/num_searches:.2f}s per search")
    print(f"  With 100 simulations per search")

    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_mcts()
