# Chess AI Project

This project implements a chess AI using deep reinforcement learning and Monte Carlo Tree Search (MCTS). It includes components for training the AI, playing games against it, and analyzing game results.

## Files and Their Purposes

1. `model.py`: Defines the neural network architecture (ChessNet) used by the AI.
2. `chess_env.py`: Implements the chess environment using the python-chess library.
3. `agent.py`: Defines the ChessAgent class, which uses the neural network to make decisions.
4. `train.py`: Main script for training the chess AI.
5. `play.py`: Script to play games using trained models.
6. `analyze.py`: Script to analyze saved games and generate statistics.
7. `cuda-test.py`: Script to check CUDA availability and GPU information.

## Setup and Installation

1. Ensure you have Python 3.7+ and pip installed.

2. Clone this repository:
   ```
   git clone https://github.com/yourusername/chess-ai.git
   cd chess-ai
   ```

3. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install PyTorch with CUDA support (if you have a compatible GPU):
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Note: This installs PyTorch with CUDA 11.8 support. Check the official PyTorch website for the appropriate command for your CUDA version.

5. Install other required packages:
   ```
   pip install -r requirements.txt
   ```

6. Verify the installation by running the CUDA test script:
   ```
   python cuda-test.py
   ```
   This will output information about CUDA availability and your GPU(s).

## Usage

### Checking CUDA Availability

Run the CUDA test script:
```
python cuda-test.py
```
This will output information about CUDA availability and your GPU(s).

### Training the AI

Run the training script:
```
python train.py [--episodes EPISODES] [--batch_size BATCH_SIZE] [--lr LEARNING_RATE]
```

Options:
- `--episodes`: Number of training episodes (default: 100000)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)

Example:
```
python train.py --episodes 500000 --batch_size 128 --lr 0.0005
```

This will start the training process. The script will print updates every 100 episodes, including:
- Current episode number
- Move history of the last game
- Result of the last game
- Evaluation of current models (based on 100 games played between white and black)

The script will save model checkpoints periodically.

### Playing Games

To play games using trained models:
```
python play.py [--white_model PATH] [--black_model PATH] [--mcts_simulations SIMS]
```

Options:
- `--white_model`: Path to the white player's model (default: 'white_model_100000.pth')
- `--black_model`: Path to the black player's model (default: 'black_model_100000.pth')
- `--mcts_simulations`: Number of MCTS simulations per move (default: 800)

Example:
```
python play.py --white_model latest_white.pth --black_model latest_black.pth --mcts_simulations 1000
```

You will be prompted to choose a mode:
- `ai_vs_ai`: Watch two AI agents play against each other.
- `human_vs_ai`: Play as White against the AI.
- `ai_vs_human`: Play as Black against the AI.

If playing as a human, input your moves in UCI format (e.g., "e2e4") when prompted.

### Analyzing Games

To analyze saved games:
```
python analyze.py --games_dir DIRECTORY [--num_games NUM] [--output FILE]
```

Options:
- `--games_dir`: Directory containing PGN files (required)
- `--num_games`: Number of games to analyze (default: all games in directory)
- `--output`: Output JSON file (default: 'stats.json')

Example:
```
python analyze.py --games_dir ./saved_games --num_games 1000 --output recent_stats.json
```

This will analyze up to 1000 games from the ./saved_games directory and output the results to recent_stats.json.

## Troubleshooting

- If you encounter CUDA-related errors, ensure your GPU drivers and CUDA toolkit are properly installed and compatible with the PyTorch version you're using.
- Make sure all script files and trained models are in the same directory or properly referenced when running commands.

## Future Improvements

- Implement distributed training for faster learning
- Add support for different board representations (e.g., bitboards)
- Implement an opening book to improve early game play
- Create a GUI for easier interaction with the AI during gameplay