from chess_env import ChessEnv
from agent import ChessAgent
import argparse
import chess
import chess.pgn
import torch
import random
import logging
from collections import deque

def self_play_game(white_agent, black_agent, env, episode, max_moves=200):
    state = env.reset()
    total_reward = 0
    for move_count in range(max_moves):
        if env.board.is_game_over():
            break
        
        current_player = white_agent if env.board.turn == chess.WHITE else black_agent
        current_player.update_board(env.board)
        legal_moves = env.get_legal_moves()
        
        action = current_player.select_action(state, legal_moves)
        next_state, reward, done, info = env.step(action)
        
        total_reward += abs(reward)
        current_player.store_transition(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            break

    result = env.get_result()
    if result is None:
        result = "1/2-1/2 (Incomplete Game)"

    actual_move_count = len(env.board.move_stack)
    
    return result, env.get_pgn(white_agent, black_agent, episode), actual_move_count, total_reward

def evaluate(white_agent, black_agent, num_games=100):
    env = ChessEnv()
    white_wins = 0
    black_wins = 0
    draws = 0

    for episode in range(num_games):
        result, _, _, _ = self_play_game(white_agent, black_agent, env, episode)
        if result.startswith("1-0"):
            white_wins += 1
        elif result.startswith("0-1"):
            black_wins += 1
        else:
            draws += 1

    return white_wins, black_wins, draws

def train(num_episodes, white_model_path=None, black_model_path=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    env = ChessEnv()
    white_agent = ChessAgent(chess.WHITE, initial_epsilon=0.9, epsilon_decay=0.99995, min_epsilon=0.05, lr=args.lr)
    black_agent = ChessAgent(chess.BLACK, initial_epsilon=0.9, epsilon_decay=0.99995, min_epsilon=0.05, lr=args.lr)

    if white_model_path:
        white_agent.load_model(white_model_path)
    else:
        white_agent.initialize_model_file('white')

    if black_model_path:
        black_agent.load_model(black_model_path)
    else:
        black_agent.initialize_model_file('black')

    # Metrics tracking
    window_size = 1000
    game_lengths = deque(maxlen=window_size)
    white_win_rates = deque(maxlen=window_size)
    black_win_rates = deque(maxlen=window_size)
    draw_rates = deque(maxlen=window_size)
    avg_rewards = deque(maxlen=window_size)

    for episode in range(num_episodes):
        result, move_history, actual_move_count, total_reward = self_play_game(white_agent, black_agent, env, episode)
        
        white_grad_norm = white_agent.update(args.batch_size)
        black_grad_norm = black_agent.update(args.batch_size)

        # Update epsilon
        white_agent.update_epsilon()
        black_agent.update_epsilon()

        # Update metrics
        game_lengths.append(actual_move_count)
        white_win_rates.append(1 if result.startswith("1-0") else 0)
        black_win_rates.append(1 if result.startswith("0-1") else 0)
        draw_rates.append(1 if result.startswith("1/2-1/2") else 0)
        avg_rewards.append(total_reward / actual_move_count if actual_move_count > 0 else 0)

        if episode % args.log_interval == 0:
            logger.info(f"Episode {episode}")
            logger.info(f"Game result: {result}")
            logger.info(f"Move count: {actual_move_count}")
            logger.info(f"Epsilon: White {white_agent.epsilon:.4f}, Black {black_agent.epsilon:.4f}")
            logger.info(f"Performance metrics (last {args.log_interval} games):")
            logger.info(f"  Avg game length: {sum(game_lengths) / len(game_lengths):.2f}")
            logger.info(f"  Win rates: White {sum(white_win_rates) / len(white_win_rates):.2f}, Black {sum(black_win_rates) / len(black_win_rates):.2f}")
            logger.info(f"  Draw rate: {sum(draw_rates) / len(draw_rates):.2f}")
            logger.info(f"  Avg reward per move: {sum(avg_rewards) / len(avg_rewards):.4f}")
            logger.info(f"Training metrics:")
            logger.info(f"  Gradient norm: White {white_grad_norm:.4f}, Black {black_grad_norm:.4f}")
            
            # Evaluate agents
            eval_white_wins, eval_black_wins, eval_draws = evaluate(white_agent, black_agent)
            logger.info(f"Evaluation (100 games): White wins: {eval_white_wins}, Black wins: {eval_black_wins}, Draws: {eval_draws}")
            
            # Output PGN on separate lines
            logger.info("Game PGN:")
            for line in pgn.split('\n'):
                logger.info(line)
            
            logger.info("")  # Add an empty line for better readability between episodes

        # Save models periodically
        if episode % args.save_interval == 0:
            white_agent.save_model()
            black_agent.save_model()

    logger.info("Training completed.")
    
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