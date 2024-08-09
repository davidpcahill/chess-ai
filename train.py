from chess_env import ChessEnv
from agent import ChessAgent
import argparse
import chess
import chess.pgn
import torch
import random

def self_play_game(white_agent, black_agent, env, episode=None):
    state = env.reset()
    done = False
    while not done:
        legal_moves = env.get_legal_moves()
        if env.board.turn == chess.WHITE:
            action = white_agent.select_action(state, legal_moves)
        else:
            action = black_agent.select_action(state, legal_moves)
        
        next_state, reward, done, _ = env.step(action)
        
        if env.board.turn == chess.WHITE:
            white_agent.store_transition(state, action, reward, next_state, done)
        else:
            black_agent.store_transition(state, action, -reward, next_state, done)
        
        state = next_state

    return env.get_result(), env.get_pgn(white_agent, black_agent, episode)

def evaluate(white_agent, black_agent, num_games=100):
    env = ChessEnv()
    white_wins = 0
    black_wins = 0
    draws = 0

    for _ in range(num_games):
        result, _ = self_play_game(white_agent, black_agent, env)
        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1

    return white_wins, black_wins, draws

def train(num_episodes, white_model_path=None, black_model_path=None):
    env = ChessEnv()
    white_agent = ChessAgent(chess.WHITE, initial_epsilon=0.9, epsilon_decay=0.99995, min_epsilon=0.05)
    black_agent = ChessAgent(chess.BLACK, initial_epsilon=0.9, epsilon_decay=0.99995, min_epsilon=0.05)

    if white_model_path:
        white_agent.load_model(white_model_path)
    else:
        white_agent.initialize_model_file('white')

    if black_model_path:
        black_agent.load_model(black_model_path)
    else:
        black_agent.initialize_model_file('black')

    for episode in range(num_episodes):
        result, move_history = self_play_game(white_agent, black_agent, env, episode)
        
        white_agent.update()
        black_agent.update()

        # Update epsilon
        white_agent.update_epsilon()
        black_agent.update_epsilon()

        if episode % args.log_interval == 0:
            print(f"Episode {episode}")
            print(f"Move history: {move_history}")
            print(f"Game result: {result}")
            print(f"White epsilon: {white_agent.epsilon:.4f}")
            print(f"Black epsilon: {black_agent.epsilon:.4f}")
            
            white_wins, black_wins, draws = evaluate(white_agent, black_agent)
            print(f"Evaluation: White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
            print()

        # Save models periodically
        if episode % args.save_interval == 0:
            white_agent.save_model()
            black_agent.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chess AI")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging game results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--white_model", type=str, help="Path to existing white model to resume training")
    parser.add_argument("--black_model", type=str, help="Path to existing black model to resume training")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval for saving models")
    
    args = parser.parse_args()
    
    train(args.episodes, args.white_model, args.black_model)