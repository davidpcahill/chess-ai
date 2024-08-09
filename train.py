from chess_env import ChessEnv
from agent import ChessAgent
import chess
import torch

def self_play_game(white_agent, black_agent, env):
    state = env.reset()
    done = False
    while not done:
        if env.board.turn == chess.WHITE:
            action = white_agent.select_action(state, env.get_legal_moves())
        else:
            action = black_agent.select_action(state, env.get_legal_moves())
        
        next_state, reward, done, _ = env.step(action)
        
        if env.board.turn == chess.WHITE:
            white_agent.store_transition(state, action, reward, next_state, done)
        else:
            black_agent.store_transition(state, action, -reward, next_state, done)
        
        state = next_state

    return env.get_result(), env.get_move_history()

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

def train(num_episodes):
    env = ChessEnv()
    white_agent = ChessAgent(chess.WHITE)
    black_agent = ChessAgent(chess.BLACK)

    for episode in range(num_episodes):
        result, move_history = self_play_game(white_agent, black_agent, env)
        
        white_agent.update()
        black_agent.update()

        if episode % 100 == 0:
            print(f"Episode {episode}")
            print(f"Move history: {move_history}")
            print(f"Game result: {result}")
            
            white_wins, black_wins, draws = evaluate(white_agent, black_agent)
            print(f"Evaluation: White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
            print()

        # Save models periodically
        if episode % 1000 == 0:
            torch.save(white_agent.model.state_dict(), f'white_model_{episode}.pth')
            torch.save(black_agent.model.state_dict(), f'black_model_{episode}.pth')

if __name__ == "__main__":
    train(100000)