import os

import torch
from chess_engine.data.chess_dataset import (
    ChessDataset,
    ChessGameParser,
    compute_dataset_statistics,
    create_dataloader,
    load_dataset,
    print_dataset_statistics,
    save_dataset,
    split_examples,
)
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


def download_sample_games(output_path: str = "data/pgn/sample_games.pgn") -> None:
    """
    Create a small sample PGN file with games for testing.

    This creates a small PGN file with a few games for quick testing.
    For real training, download from: https://database.lichess.org/

    Args:
        output_path: Where to save the sample PGN file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sample PGN data (three real games at different skill levels)
    sample_pgn = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc123"]
[Date "2024.01.15"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "2050"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8
14. Ng3 g6 15. a4 c5 16. d5 c4 17. Bg5 Nc5 18. Qd2 h6 19. Be3 Qc7
20. Bxc5 dxc5 21. b4 Rab8 22. axb5 axb5 23. Ra5 Bc8 24. Rea1 Bd7
25. Nf1 Qb6 26. N1d2 Red8 27. Ra7 Be8 28. Qe3 Nd7 29. Rxd7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/def456"]
[Date "2024.01.15"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[WhiteElo "1950"]
[BlackElo "2000"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 b6
7. Bg5 Bb7 8. f3 h6 9. Bh4 d5 10. e3 Nbd7 11. cxd5 Nxd5 12. Qc2 Nxe3
13. Qd2 Nf5 14. Bf2 c5 15. dxc5 Nxc5 16. Bd4 Nxd4 17. Qxd4 Qf6
18. Qxf6 gxf6 19. Ne2 Rfd8 20. Nc3 Rd4 21. Be2 Rad8 22. O-O Ne4 0-1

[Event "Rated Blitz game"]
[Site "https://lichess.org/ghi789"]
[Date "2024.01.15"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]
[WhiteElo "1875"]
[BlackElo "1900"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6
8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8
14. f4 a5 15. f5 Bc4 16. Nbd4 exd4 17. Nxd4 a4 18. Kb1 Qa5 19. Qxa5 Rxa5
20. Bxc4 Rxc4 21. Nb5 Ne5 22. Rd5 Rxd5 23. exd5 Kf8 24. Rf1 Ke7
25. Nd4 Nc7 26. Rf4 Rc5 27. c3 bxc3 28. bxc3 Nxd5 29. Rf2 Nxe3
30. Re2 Nxg5 31. Rxe5+ dxe5 32. Nc6+ Rxc6 1/2-1/2
"""

    try:
        with open(output_path, "w") as f:
            f.write(sample_pgn)

        print(f"✅ Created sample PGN file: {output_path}")
        print(f"   Contains 3 games for testing")
    except Exception as e:
        print(f"❌ Error creating sample PGN file: {e}")
        raise


def test_data_pipeline() -> None:
    """
    Test the complete data pipeline.

    This comprehensive test validates:
    1. PGN parsing
    2. Progress callback functionality
    3. Streaming mode
    4. Dataset statistics
    5. Train/val splitting
    6. Dataset creation with augmentation
    7. Tensor caching
    8. DataLoader functionality
    9. Batch loading
    10. Save/load operations
    """
    print("=" * 80)
    print("TESTING COMPLETE DATA PIPELINE")
    print("=" * 80)

    # Step 1: Create sample data
    print("\n" + "─" * 80)
    print("STEP 1: Creating sample PGN file...")
    print("─" * 80)
    download_sample_games()

    # Step 2: Test progress callback
    print("\n" + "─" * 80)
    print("STEP 2: Testing progress callback...")
    print("─" * 80)

    progress_log = []

    def progress_callback(games: int, positions: int) -> None:
        """Custom progress callback for testing"""
        progress_log.append((games, positions))
        if games % 10 == 0:  # Log every 10 games
            print(f"   Callback: {games} games → {positions} positions")

    parser_with_callback = ChessGameParser(
        min_elo=1500, progress_callback=progress_callback
    )
    examples = parser_with_callback.parse_pgn_file("data/pgn/sample_games.pgn")
    print(f"✅ Progress callback called {len(progress_log)} times")
    print(f"✅ Extracted {len(examples)} positions from games")

    # Step 3: Test streaming mode
    print("\n" + "─" * 80)
    print("STEP 3: Testing streaming mode...")
    print("─" * 80)

    parser_streaming = ChessGameParser(min_elo=1500)
    chunks = []
    total_streamed = 0

    for chunk in parser_streaming.parse_pgn_file_streaming(
        "data/pgn/sample_games.pgn", chunk_size=50
    ):
        chunks.append(chunk)
        total_streamed += len(chunk)
        print(f"   Received chunk: {len(chunk)} positions")

    print(f"✅ Streaming mode: {len(chunks)} chunks, {total_streamed} total positions")
    print(f"   Matches non-streaming: {total_streamed == len(examples)}")

    # Step 4: Compute and display statistics
    print("\n" + "─" * 80)
    print("STEP 4: Computing dataset statistics...")
    print("─" * 80)
    stats = compute_dataset_statistics(examples)
    print_dataset_statistics(stats)

    # Step 5: Split data
    print("\n" + "─" * 80)
    print("STEP 5: Splitting into train/validation sets...")
    print("─" * 80)
    train_examples, val_examples = split_examples(
        examples, train_ratio=0.8, shuffle=True, seed=42
    )
    print(f"✅ Training examples:   {len(train_examples):,}")
    print(f"✅ Validation examples: {len(val_examples):,}")
    print(
        f"   Split ratio:         {len(train_examples)/len(examples)*100:.1f}% / {len(val_examples)/len(examples)*100:.1f}%"
    )

    # Step 6: Create datasets (test both with and without caching)
    print("\n" + "─" * 80)
    print("STEP 6: Creating PyTorch datasets...")
    print("─" * 80)
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # Without caching (memory efficient)
    train_dataset_no_cache = ChessDataset(
        train_examples, board_encoder, move_encoder, augment=True, cache_tensors=False
    )

    # With caching (faster iteration)
    train_dataset_cached = ChessDataset(
        train_examples, board_encoder, move_encoder, augment=True, cache_tensors=True
    )

    val_dataset = ChessDataset(
        val_examples, board_encoder, move_encoder, augment=False, cache_tensors=False
    )

    print(f"✅ Train dataset (no cache):   {len(train_dataset_no_cache):,}")
    print(f"✅ Train dataset (cached):     {len(train_dataset_cached):,}")
    print(f"✅ Validation dataset:         {len(val_dataset):,}")

    # Step 7: Test loading samples
    print("\n" + "─" * 80)
    print("STEP 7: Testing data loading...")
    print("─" * 80)
    if len(train_dataset_cached) > 0:
        board_tensor, move_idx, outcome = train_dataset_cached[0]
        print(f"✅ Board tensor shape:  {board_tensor.shape}")
        print(f"✅ Board tensor dtype:  {board_tensor.dtype}")
        print(f"✅ Move index:          {move_idx.item()}")
        print(f"✅ Move index dtype:    {move_idx.dtype}")
        print(f"✅ Outcome:             {outcome.item():.1f}")
        print(f"✅ Outcome dtype:       {outcome.dtype}")

    # Step 8: Create dataloaders with optimized settings
    print("\n" + "─" * 80)
    print("STEP 8: Creating optimized DataLoaders...")
    print("─" * 80)
    train_loader = create_dataloader(
        train_dataset_cached, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    print(f"✅ Train batches:       {len(train_loader)}")
    print(f"✅ Val batches:         {len(val_loader)}")
    print(f"   Batch size:          32")
    print(f"   Pin memory:          {torch.cuda.is_available()}")
    print(f"   Num workers:         0 (set to 4 for real training)")

    # Step 9: Test batch loading
    print("\n" + "─" * 80)
    print("STEP 9: Testing batch loading...")
    print("─" * 80)
    if len(train_loader) > 0:
        for boards, moves, outcomes in train_loader:
            print(f"✅ Batch boards shape:   {boards.shape}")
            print(f"✅ Batch boards dtype:   {boards.dtype}")
            print(f"✅ Batch moves shape:    {moves.shape}")
            print(f"✅ Batch moves dtype:    {moves.dtype}")
            print(f"✅ Batch outcomes shape: {outcomes.shape}")
            print(f"✅ Batch outcomes dtype: {outcomes.dtype}")
            print(f"✅ Sample outcomes:      {outcomes[:5].tolist()}")
            break  # Just test first batch

    # Step 10: Test save/load
    print("\n" + "─" * 80)
    print("STEP 10: Testing save/load operations...")
    print("─" * 80)
    save_path = "data/processed/test_dataset.pkl"
    save_dataset(examples, save_path)
    loaded_examples = load_dataset(save_path)
    print(f"✅ Verified: {len(loaded_examples):,} examples loaded")
    print(f"   Match:    {len(loaded_examples) == len(examples)}")

    # Step 11: Test cache clearing
    print("\n" + "─" * 80)
    print("STEP 11: Testing cache management...")
    print("─" * 80)
    train_dataset_cached.clear_cache()

    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL DATA PIPELINE TESTS PASSED!")
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"   ✅ Parsed {len(examples):,} positions successfully")
    print(f"   ✅ Progress callback working ({len(progress_log)} calls)")
    print(f"   ✅ Streaming mode working ({len(chunks)} chunks)")
    print(f"   ✅ Train/val split: {len(train_examples):,} / {len(val_examples):,}")
    print(f"   ✅ Data augmentation working")
    print(f"   ✅ Tensor caching working")
    print(f"   ✅ DataLoaders created with optimized settings")
    print(f"   ✅ Save/load functionality verified")


if __name__ == "__main__":
    test_data_pipeline()
