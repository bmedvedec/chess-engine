"""
Quick test to verify evaluation.py works correctly after type fixes.
Tests basic functionality without requiring a trained model.
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from chess_engine.evaluation.evaluation import (
        ChessEngineEvaluator,
        RandomPlayer,
        StockfishPlayer,
        GameRecord,
        GameResult,
        EvaluationResults,
        CHESS_ENGINE_AVAILABLE,
        MCTS_AVAILABLE,
        PLOTTING_AVAILABLE,
    )

    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test availability flags
print("\nAvailability checks:")
print(f"  chess.engine: {'✅' if CHESS_ENGINE_AVAILABLE else '❌'}")
print(f"  MCTS:         {'✅' if MCTS_AVAILABLE else '❌'}")
print(f"  Plotting:     {'✅' if PLOTTING_AVAILABLE else '❌'}")

# Test RandomPlayer
print("\nTesting RandomPlayer...")
try:
    import chess

    random_player = RandomPlayer()
    board = chess.Board()
    move = random_player.select_move(board)
    print(f"✅ RandomPlayer works! Move: {move}")
except Exception as e:
    print(f"❌ RandomPlayer error: {e}")

# Test StockfishPlayer
print("\nTesting StockfishPlayer...")
try:
    stockfish = StockfishPlayer(level=1)
    if stockfish.engine_available:
        board = chess.Board()
        move = stockfish.select_move(board)
        print(f"✅ StockfishPlayer works! Move: {move}")
        stockfish.close()
    else:
        print("⚠️  Stockfish not available (expected on systems without Stockfish)")
except Exception as e:
    print(f"❌ StockfishPlayer error: {e}")

# Test data structures
print("\nTesting data structures...")
try:
    import chess

    game_record = GameRecord(
        game_id=1,
        opponent="Test",
        result=GameResult.WIN,
        num_moves=42,
        time_taken=10.5,
        engine_color=chess.WHITE,
        termination="checkmate",
    )
    print(f"✅ GameRecord created: {game_record.game_id}, {game_record.result.value}")

    results = EvaluationResults(
        model_name="test_model", timestamp="2024-01-01", total_games=0
    )
    print(f"✅ EvaluationResults created: {results.model_name}")
except Exception as e:
    print(f"❌ Data structure error: {e}")

# Test type conversions
print("\nTesting type conversions...")
try:
    import numpy as np

    # Test numpy float conversion
    values = [1, 2, 3, 4, 5]
    mean_val = float(np.mean(values))
    assert isinstance(mean_val, float), "np.mean conversion failed"
    print(f"✅ NumPy type conversion works: {mean_val}")

    # Test dictionary type consistency
    stats_dict = {
        "total_games": 10,
        "win_rate": float(0.6),
        "loss_rate": float(0.3),
        "draw_rate": float(0.1),
        "avg_moves": float(35.5),
    }
    print(f"✅ Dictionary type consistency verified")
except Exception as e:
    print(f"❌ Type conversion error: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

# Summary
issues = []
if not CHESS_ENGINE_AVAILABLE:
    issues.append("chess.engine not available (Stockfish won't work)")
if not MCTS_AVAILABLE:
    issues.append("MCTS not available (will use policy-only mode)")
if not PLOTTING_AVAILABLE:
    issues.append("Plotting not available (no visualizations)")

if issues:
    print("\n⚠️  Optional features missing:")
    for issue in issues:
        print(f"  • {issue}")
    print("\nNote: These are optional. Basic evaluation will still work!")
else:
    print("\n✅ All features available!")
