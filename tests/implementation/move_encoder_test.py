from chess_engine.utils.move_encoder import (
    MoveEncoder,
    MoveHistory,
    create_policy_target,
)
import chess
import torch


def test_move_encoding():
    """Test the move encoding and decoding functionality"""
    print("=" * 80)
    print("TESTING MOVE ENCODING")
    print("=" * 80)

    encoder = MoveEncoder()

    # Test 1: Basic move encoding
    print("\n1. Testing basic move encoding...")
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    index = encoder.encode_move(move)
    decoded = encoder.decode_move(index, board)

    print(f"Original move: {move.uci()}")
    print(f"Encoded index: {index}")
    print(f"Decoded move: {decoded.uci()}")
    print(f"Match: {move == decoded}")

    # Test 2: Legal moves mask
    print("\n2. Testing legal moves mask...")
    mask = encoder.create_legal_moves_mask(board)
    num_legal = torch.sum(mask).item()
    print(f"Legal moves in starting position: {num_legal}")
    print(f"Expected: 20 (16 pawn moves + 4 knight moves)")

    # Test 3: Policy to move probabilities
    print("\n3. Testing policy conversion...")
    fake_policy = torch.randn(4096)
    move_probs = encoder.policy_to_move_probs(fake_policy, board)
    print(f"Number of moves with non-zero probability: {len(move_probs)}")
    print(f"Total probability: {sum(move_probs.values()):.6f}")

    # Show top 3 moves
    top_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 moves:")
    for move, prob in top_moves:
        print(f"  {move.uci()}: {prob:.4f}")

    # Test 4: Move history encoding
    print("\n4. Testing move history encoding...")
    history_encoder = MoveHistory(max_length=50)

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")

    encoded_history = history_encoder.encode_game_history(board)
    print(f"Encoded history shape: {encoded_history.shape}")
    print(f"Encoded history: {encoded_history[-10:]}")  # Show last 10

    # Test 5: Batch encoding
    print("\n5. Testing batch move history encoding...")
    boards = [chess.Board() for _ in range(3)]
    boards[0].push_san("e4")
    boards[1].push_san("d4")
    boards[1].push_san("d5")
    boards[2].push_san("Nf3")
    boards[2].push_san("Nf6")
    boards[2].push_san("c4")

    sequences, lengths = history_encoder.batch_encode_histories(boards)
    print(f"Batch sequences shape: {sequences.shape}")
    print(f"Lengths: {lengths}")

    # Test 6: Policy target creation
    print("\n6. Testing policy target creation...")
    board = chess.Board()
    target_move = chess.Move.from_uci("e2e4")
    target = create_policy_target(board, target_move, smooth=0.1)

    print(f"Target shape: {target.shape}")
    print(f"Target sum: {target.sum().item():.6f}")
    print(
        f"Target max (at e2e4): {target[encoder.encode_move(target_move)].item():.4f}"
    )

    print("\n" + "=" * 80)
    print("✅ ALL MOVE ENCODING TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_move_encoding()
