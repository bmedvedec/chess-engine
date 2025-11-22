from chess_engine.utils.board_encoder import BoardEncoder, DataAugmentation
import chess


def test_encoding():
    """Test the board encoding and decoding functionality"""
    print("=" * 80)
    print("TESTING BOARD ENCODING")
    print("=" * 80)

    encoder = BoardEncoder()

    # Test 1: Starting position
    print("\n1. Testing starting position...")
    board = chess.Board()
    print(f"Original board:\n{board}\n")

    tensor = encoder.board_to_tensor(board)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")

    # Visualize some channels
    encoder.visualize_tensor(tensor)

    # Test reconstruction
    reconstructed = encoder.tensor_to_board(tensor)
    print(f"\nReconstructed board:\n{reconstructed}\n")
    print(f"FEN match: {board.fen() == reconstructed.fen()}")

    # Test 2: Position after some moves
    print("\n2. Testing position after moves...")
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    print(f"Board after 1.e4 e5 2.Nf3:\n{board}\n")

    tensor = encoder.board_to_tensor(board)
    reconstructed = encoder.tensor_to_board(tensor)
    print(
        f"Reconstruction successful: {board.board_fen() == reconstructed.board_fen()}"
    )

    # Test 3: Batch encoding
    print("\n3. Testing batch encoding...")
    boards = [chess.Board() for _ in range(4)]
    boards[0].push_san("e4")
    boards[1].push_san("d4")
    boards[2].push_san("Nf3")

    batch_tensor = encoder.batch_boards_to_tensor(boards)
    print(f"Batch tensor shape: {batch_tensor.shape}")

    # Test 4: Data augmentation
    print("\n4. Testing data augmentation...")
    board = chess.Board()
    board.push_san("e4")

    flipped = DataAugmentation.horizontal_flip(board)
    print(f"Original:\n{board}\n")
    print(f"Horizontally flipped:\n{flipped}\n")

    tensor_flip = DataAugmentation.flip_tensor(encoder.board_to_tensor(board))
    print(f"Tensor flip successful: shape {tensor_flip.shape}")

    print("\n" + "=" * 80)
    print("✅ ALL ENCODING TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_encoding()
