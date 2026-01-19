"""
Configuration file for chess engine project
"""

import torch
from pathlib import Path
import multiprocessing

# =============================================================================
# HARDWARE DETECTION
# =============================================================================


def get_hardware_info():
    """Get hardware information"""
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": multiprocessing.cpu_count(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )
        info["cuda_version"] = torch.version.cuda
        info["compute_capability"] = torch.cuda.get_device_capability(0)

    return info


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HARDWARE CONFIGURATION (GTX 1070 Ti + Ryzen 7 7800X3D)
# =============================================================================

HARDWARE_CONFIG = {
    # GPU Settings (GTX 1070 Ti - 8GB VRAM, Pascal architecture)
    "device": DEVICE,
    "mixed_precision": True,  # ✅ Enable AMP (20-30% memory savings)
    "pin_memory": True,  # Faster data transfer to GPU
    "non_blocking": True,  # Async data transfer
    # Batch sizes optimized for 8GB VRAM with mixed precision
    # With AMP, you can use larger batches (1.3-1.5x normal size)
    "train_batch_size": 256,  # Start here, can go to 384 with AMP
    "eval_batch_size": 512,  # Evaluation uses less memory
    "mcts_batch_size": 16,  # For batched NN evaluations in MCTS
    # DataLoader workers (Ryzen 7 7800X3D - 8 cores/16 threads)
    "num_workers": 4,  # 1/4 of CPU threads for data loading
    "prefetch_factor": 2,  # Prefetch 2 batches per worker
    # Self-play parallelization (Ryzen 7 7800X3D)
    "num_self_play_workers": 12,  # 12 parallel workers (leave 4 for system)
    "self_play_games_per_worker": 40,  # Each worker generates ~40 games
    # Model size (for 8GB VRAM with mixed precision)
    "num_residual_blocks": 10,  # Can go to 12-15 with AMP
    "num_filters": 256,  # Channel width
    # Memory management
    "gradient_accumulation_steps": 1,  # Increase if still OOM
    "empty_cache_interval": 10,  # Clear CUDA cache every N batches
    "max_grad_norm": 1.0,  # Gradient clipping threshold
    # Training optimizations
    "cudnn_benchmark": True,  # Enable cuDNN auto-tuner (10-20% speedup)
    "cudnn_deterministic": False,  # Disable for better performance
    # Monitoring
    "log_gpu_stats": True,  # Log GPU usage during training
    "gpu_memory_fraction": 0.95,  # Use up to 95% of GPU memory
}

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

BOARD_CHANNELS = 20  # Input channels for CNN
CNN_FILTERS = HARDWARE_CONFIG["num_filters"]  # Number of filters in CNN
CNN_BLOCKS = HARDWARE_CONFIG["num_residual_blocks"]  # Number of residual blocks
MOVE_EMBEDDING_DIM = 64
RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = HARDWARE_CONFIG["train_batch_size"]
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
NUM_WORKERS = HARDWARE_CONFIG["num_workers"]

# Mixed precision training
USE_AMP = HARDWARE_CONFIG["mixed_precision"]

# =============================================================================
# MCTS PARAMETERS
# =============================================================================

MCTS_SIMULATIONS = 800
C_PUCT = 1.5
TEMPERATURE = 1.0

# =============================================================================
# RL TRAINING
# =============================================================================

SELF_PLAY_GAMES = 100
TRAINING_STEPS = 1000
REPLAY_BUFFER_SIZE = 500000  # Large buffer (32GB RAM can handle this)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def setup_training_device():
    """
    Setup and configure the training device with optimal settings.
    Call this at the start of your training script.

    Returns:
        torch.device: Configured device
    """
    device = torch.device(HARDWARE_CONFIG["device"])

    if device.type == "cuda":
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = HARDWARE_CONFIG["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = HARDWARE_CONFIG["cudnn_deterministic"]

        # Set memory fraction if specified
        if HARDWARE_CONFIG.get("gpu_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(
                HARDWARE_CONFIG["gpu_memory_fraction"]
            )

        # Print GPU info
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )
        print(
            f"⚡ Mixed Precision (AMP): {'Enabled' if HARDWARE_CONFIG['mixed_precision'] else 'Disabled'}"
        )
        print(f"🔧 CUDA Version: {torch.version.cuda}")

        # Check compute capability
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"📊 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

        # GTX 1070 Ti is Pascal (6.1) - confirm AMP support
        if compute_cap[0] >= 6:
            print(f"✅ AMP supported (Compute Capability >= 6.0)")
        else:
            print(f"⚠️  AMP may not work optimally (Compute Capability < 6.0)")
            HARDWARE_CONFIG["mixed_precision"] = False
    else:
        print("⚠️  Training on CPU (slower)")
        HARDWARE_CONFIG["mixed_precision"] = False  # Disable AMP on CPU

    return device


def print_system_info():
    """Print complete system information for debugging"""
    import platform

    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False

    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    # OS
    print(f"\n🖥️  Operating System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")

    # CPU
    print(f"\n⚙️  CPU: {platform.processor()}")
    print(f"🔢 CPU Cores: {multiprocessing.cpu_count()} logical threads")

    if has_psutil:
        print(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )
        print(f"🔧 CUDA Version: {torch.version.cuda}")

        if torch.backends.cudnn.enabled:
            print(f"⚡ cuDNN Version: {torch.backends.cudnn.version()}")

        compute_cap = torch.cuda.get_device_capability(0)
        print(f"📊 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    else:
        print("\n⚠️  No GPU detected - training will be slow!")

    # PyTorch
    print(f"\n🔥 PyTorch Version: {torch.__version__}")
    print(
        f"🎯 Mixed Precision (AMP): {'Available' if torch.cuda.is_available() else 'Not Available (CPU)'}"
    )

    print("=" * 70)


def print_hardware_config():
    """Print current hardware configuration"""
    print("\n" + "=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)

    for key, value in HARDWARE_CONFIG.items():
        print(f"{key:30s}: {value}")

    print("=" * 70)


# Print configuration on import
print(f"Configuration loaded. Using device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(
        f"Mixed Precision: {'Enabled' if HARDWARE_CONFIG['mixed_precision'] else 'Disabled'}"
    )
