from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.board_decoder import tensor_to_board
from chess_engine.utils.augmentations import DataAugmentation
from chess_engine.utils.visualization import visualize_tensor
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
    visualize_tensor(tensor)

    # Test reconstruction
    reconstructed = tensor_to_board(tensor)
    print(f"\nReconstructed board:\n{reconstructed}\n")
    print(f"FEN match: {board.fen() == reconstructed.fen()}")

    # Test 2: Position after some moves
    print("\n2. Testing position after moves...")
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    print(f"Board after 1.e4 e5 2.Nf3:\n{board}\n")

    tensor = encoder.board_to_tensor(board)
    reconstructed = tensor_to_board(tensor)
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

    # Test 5: Move flipping
    print("\n5. Testing move flipping (NEW)...")
    board = chess.Board()

    # Test a few different moves
    test_moves = [
        chess.Move.from_uci("e2e4"),  # e-file move
        chess.Move.from_uci("a2a4"),  # a-file move (should flip to h-file)
        chess.Move.from_uci("g1f3"),  # Knight move
        chess.Move.from_uci("b1c3"),  # Knight move
    ]

    print("\nOriginal moves -> Flipped moves:")
    for move in test_moves:
        flipped_move = DataAugmentation.flip_move(move)
        print(f"  {move.uci():6s} -> {flipped_move.uci()}")

        # Verify the flip is correct
        from_file_orig = chess.square_file(move.from_square)
        to_file_orig = chess.square_file(move.to_square)
        from_file_flip = chess.square_file(flipped_move.from_square)
        to_file_flip = chess.square_file(flipped_move.to_square)

        # Files should be mirrored: 0↔7, 1↔6, 2↔5, 3↔4
        assert from_file_flip == 7 - from_file_orig, f"From file not flipped correctly"
        assert to_file_flip == 7 - to_file_orig, f"To file not flipped correctly"

        # Ranks should stay the same
        from_rank_orig = chess.square_rank(move.from_square)
        to_rank_orig = chess.square_rank(move.to_square)
        from_rank_flip = chess.square_rank(flipped_move.from_square)
        to_rank_flip = chess.square_rank(flipped_move.to_square)

        assert from_rank_flip == from_rank_orig, f"From rank changed unexpectedly"
        assert to_rank_flip == to_rank_orig, f"To rank changed unexpectedly"

    print("[OK] All move flips verified correct!")

    # Test 6: Move index flipping (4672 AlphaZero encoding)
    print("\n6. Testing move index flipping (NEW)...")

    from chess_engine.utils.move_encoder import MoveEncoder as _MoveEnc

    _enc = _MoveEnc()

    # Use valid 4672-encoded moves (updated from old 4096 raw indices)
    test_moves_6 = [
        chess.Move.from_uci("e2e4"),  # Common pawn move
        chess.Move.from_uci("a2a4"),  # a-file pawn (should flip to h-file)
        chess.Move.from_uci("h2h4"),  # h-file pawn (should flip to a-file)
        chess.Move.from_uci("g1f3"),  # Knight move
        chess.Move.from_uci("b1c3"),  # Knight move
    ]

    print("\nOriginal index -> Flipped index:")
    for move in test_moves_6:
        move_index = _enc.encode_move(move)
        flipped_index = DataAugmentation.flip_move_index(move_index)
        flipped_move = _enc.decode_move(flipped_index)

        print(
            f"  {move.uci():6s} (idx {move_index:4d}) -> {flipped_move.uci():6s} (idx {flipped_index:4d})"
        )

        # Verify files are flipped
        from_file_orig = chess.square_file(move.from_square)
        to_file_orig = chess.square_file(move.to_square)
        from_file_flip = chess.square_file(flipped_move.from_square)
        to_file_flip = chess.square_file(flipped_move.to_square)

        assert (
            from_file_flip == 7 - from_file_orig
        ), f"From file not flipped for {move.uci()}"
        assert to_file_flip == 7 - to_file_orig, f"To file not flipped for {move.uci()}"

    print("[OK] All move index flips verified correct!")

    # Test 7: Consistency between move and move_index flipping
    print("\n7. Testing consistency between flip_move and flip_move_index...")

    from chess_engine.utils.move_encoder import MoveEncoder

    move_encoder = MoveEncoder()

    board = chess.Board()
    test_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b1c3"),
    ]

    for move in test_moves:
        # Method 1: Flip the move object, then encode
        flipped_move = DataAugmentation.flip_move(move)
        flipped_index_method1 = move_encoder.encode_move(flipped_move)

        # Method 2: Encode the move, then flip the index
        original_index = move_encoder.encode_move(move)
        flipped_index_method2 = DataAugmentation.flip_move_index(original_index)

        # Both methods should give the same result
        print(
            f"  {move.uci()}: method1={flipped_index_method1}, method2={flipped_index_method2}"
        )
        assert (
            flipped_index_method1 == flipped_index_method2
        ), f"Inconsistent flipping for {move.uci()}"

    print("[OK] flip_move and flip_move_index are consistent!")

    # Test 8: Round-trip test (flip twice = original)
    print("\n8. Testing round-trip (flip twice = original)...")

    board = chess.Board()
    test_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("a2a4"),
        chess.Move.from_uci("h2h4"),
    ]

    for move in test_moves:
        # Flip twice should give original
        flipped_once = DataAugmentation.flip_move(move)
        flipped_twice = DataAugmentation.flip_move(flipped_once)

        assert move == flipped_twice, f"Round-trip failed for {move.uci()}"
        print(f"  {move.uci()} -> {flipped_once.uci()} -> {flipped_twice.uci()} [OK]")

    # Same for move indices
    for move in test_moves:
        original_index = move_encoder.encode_move(move)
        flipped_once = DataAugmentation.flip_move_index(original_index)
        flipped_twice = DataAugmentation.flip_move_index(flipped_once)

        assert (
            original_index == flipped_twice
        ), f"Round-trip failed for index {original_index}"

    print("[OK] Round-trip test passed!")

    # Test 9: Promotion moves
    print("\n9. Testing promotion move flipping...")

    # Create a position where white pawn can promote
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promotion_move = chess.Move.from_uci("a7a8q")  # Promote to queen

    flipped_promotion = DataAugmentation.flip_move(promotion_move)
    print(f"  Original promotion:  {promotion_move.uci()}")
    print(f"  Flipped promotion:   {flipped_promotion.uci()}")

    # Verify promotion piece is preserved
    assert (
        promotion_move.promotion == flipped_promotion.promotion
    ), "Promotion piece not preserved"
    print("[OK] Promotion moves handled correctly!")

    print("\n" + "=" * 80)
    print("[OK] ALL ENCODING TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_encoding()
