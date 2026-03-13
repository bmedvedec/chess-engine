import time
import torch
from typing import TypedDict

from chess_engine.models.hybrid.hybrid_net import HybridChessNet


class BenchmarkResult(TypedDict):
    avg_time_ms: float
    iterations: int
    batch_size: int
    device: str


def benchmark_inference(
    model: HybridChessNet,
    num_iterations: int = 100,
    batch_size: int = 1,
    seq_len: int = 50,
) -> BenchmarkResult:
    model.eval()
    device = model.device

    board = torch.randn(batch_size, 22, 8, 8, device=device)
    history = torch.randint(
        0, model.config.rnn_num_moves, (batch_size, seq_len), device=device
    )
    lengths = torch.full((batch_size,), seq_len, device=device)

    with torch.no_grad():
        for _ in range(10):
            model(board, history, lengths)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            model(board, history, lengths)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        "avg_time_ms": (elapsed / num_iterations) * 1000,
        "iterations": num_iterations,
        "batch_size": batch_size,
        "device": str(device),
    }
