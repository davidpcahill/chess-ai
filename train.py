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
    for move_count in range(max_moves):
        if env.board.is_game_over():
            break
        
        current_player = white_agent if env.board.turn == chess.WHITE else black_agent
        current_player.update_board(env.board)
        legal_moves = env.get_legal_moves()
        
        illegal_move_attempts = 0
        while True:
            action = current_player.select_action(state, legal_moves)
            next_state, reward, done, info = env.step(action)
            
            if info.get("illegal_move"):
                current_player.update_illegal_move(state, action)
                illegal_move_attempts += 1
                if illegal_move_attempts > 10:  # Prevent infinite loops
                    done = True
                    break
                continue
            else:
                break
        
        current_player.store_transition(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            break

    # Force end of game if max_moves is reached
    if move_count == max_moves - 1:
        done = True
        result = "1/2-1/2"  # Draw if max moves reached
    else:
        result = env.get_result()

    if result is None:
        result = "1/2-1/2"  # Change '*' to '1/2-1/2' for incomplete games

    actual_move_count = len(env.board.move_stack)
    
    return result, env.get_pgn(white_agent, black_agent, episode), actual_move_count

def evaluate(white_agent, black_agent, num_games=100):
    env = ChessEnv()
    white_wins = 0
    black_wins = 0
    draws = 0

    for episode in range(num_games):
        result, _, _ = self_play_game(white_agent, black_agent, env, episode)
        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
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
    game_lengths = deque(maxlen=100)
    white_win_rates = deque(maxlen=100)
    black_win_rates = deque(maxlen=100)
    draw_rates = deque(maxlen=100)
    illegal_move_counts = deque(maxlen=100)
    avg_rewards = deque(maxlen=100)

    for episode in range(num_episodes):
        result, move_history, actual_move_count = self_play_game(white_agent, black_agent, env, episode)
        
        white_agent.update(args.batch_size)
        black_agent.update(args.batch_size)

        # Update epsilon
        white_agent.update_epsilon()
        black_agent.update_epsilon()

        # Update metrics
        game_lengths.append(actual_move_count)
        white_win_rates.append(1 if result == "1-0" else 0)
        black_win_rates.append(1 if result == "0-1" else 0)
        draw_rates.append(1 if result == "1/2-1/2" else 0)
        illegal_move_counts.append(white_agent.illegal_moves_counter + black_agent.illegal_moves_counter)

        # Safely calculate average reward
        total_rewards = white_agent.recent_rewards + black_agent.recent_rewards
        if total_rewards:
            avg_rewards.append(sum(total_rewards) / len(total_rewards))
        else:
            avg_rewards.append(0)  # Append 0 if no rewards are available

        if episode % args.log_interval == 0:
            logger.info(f"Episode {episode}")
            logger.info(f"Game result: {result}")
            logger.info(f"Move history: {move_history}")
            logger.info(f"Actual move count: {actual_move_count}")
            logger.info(f"White epsilon: {white_agent.epsilon:.4f}")
            logger.info(f"Black epsilon: {black_agent.epsilon:.4f}")
            logger.info(f"Avg game length: {sum(game_lengths) / len(game_lengths):.2f}")
            logger.info(f"White win rate: {sum(white_win_rates) / len(white_win_rates):.2f}")
            logger.info(f"Black win rate: {sum(black_win_rates) / len(black_win_rates):.2f}")
            logger.info(f"Draw rate: {sum(draw_rates) / len(draw_rates):.2f}")
            logger.info(f"Avg illegal moves per game: {sum(illegal_move_counts) / len(illegal_move_counts):.2f}")
            logger.info(f"Avg reward: {sum(avg_rewards) / len(avg_rewards):.4f}")
            
            # Log network statistics
            white_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in white_agent.model.parameters() if p.grad is not None) ** 0.5
            black_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in black_agent.model.parameters() if p.grad is not None) ** 0.5
            logger.info(f"White gradient norm: {white_grad_norm:.4f}")
            logger.info(f"Black gradient norm: {black_grad_norm:.4f}")
            
            # Evaluate agents
            eval_white_wins, eval_black_wins, eval_draws = evaluate(white_agent, black_agent)
            logger.info(f"Evaluation: White wins: {eval_white_wins}, Black wins: {eval_black_wins}, Draws: {eval_draws}")
            logger.info("")

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