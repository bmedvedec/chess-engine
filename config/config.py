"""
Configuration file for chess engine project
"""

import torch
import os

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4  # For data loading

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Model architecture
BOARD_CHANNELS = 20  # Input channels for CNN
CNN_FILTERS = 256    # Number of filters in CNN
CNN_BLOCKS = 10      # Number of residual blocks
MOVE_EMBEDDING_DIM = 64
RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 2

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# MCTS parameters
MCTS_SIMULATIONS = 800
C_PUCT = 1.5
TEMPERATURE = 1.0

# RL training
SELF_PLAY_GAMES = 100
TRAINING_STEPS = 1000
REPLAY_BUFFER_SIZE = 100000

print(f"Configuration loaded. Using device: {DEVICE}")