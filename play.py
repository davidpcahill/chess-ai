import torch
import chess
import math
import random
import numpy as np
from model import ChessNet
from chess_env import ChessEnv
from agent import ChessAgent

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board):
        root = MCTSNode(board)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.children:
                action, node = self.select_child(node)
                search_path.append(node)

            value, policy = self.evaluate(node.board)
            self.expand(node, policy)
            self.backpropagate(search_path, value)

        return self.select_action(root)

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = child.value / (child.visits + 1e-8) + \
                    self.c_puct * math.sqrt(node.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, node, policy):
        for move in node.board.legal_moves:
            action = move.uci()
            if policy[self.move_to_index(action)] > 0:
                child_board = node.board.copy()
                child_board.push_uci(action)
                node.children[action] = MCTSNode(child_board, parent=node)

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visits += 1
            node.value += value
            value = -value  # Flip the value for the opponent

    def select_action(self, root):
        visits = [child.visits for child in root.children.values()]
        actions = list(root.children.keys())
        return actions[visits.index(max(visits))]

    def evaluate(self, board):
        state = board_to_state(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            value, policy = self.model(state_tensor)
        return value.item(), policy.squeeze(0).cpu().numpy()

    def move_to_index(self, move):
        from_square = chess.SQUARE_NAMES.index(move[:2])
        to_square = chess.SQUARE_NAMES.index(move[2:4])
        return from_square * 64 + to_square

def load_model(path):
    model = ChessNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_ai_move(agent, board):
    mcts = MCTS(agent.model)
    action = mcts.search(board)
    return action

def print_board(board):
    print(board)
    print()

def get_human_move(board):
    while True:
        move = input("Enter your move (in UCI format, e.g., e2e4): ")
        try:
            if chess.Move.from_uci(move) in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid input. Please use UCI format (e.g., e2e4)")

def play_game(white_agent, black_agent):
    board = chess.Board()
    env = ChessEnv()

    human_color = input("Choose your color (white/black): ").lower()
    human_is_white = human_color == "white"

    while not board.is_game_over():
        print_board(board)
        
        if board.turn == chess.WHITE:
            if human_is_white:
                move = get_human_move(board)
            else:
                move = get_ai_move(white_agent, board)
                print(f"White AI moves: {move}")
        else:
            if not human_is_white:
                move = get_human_move(board)
            else:
                move = get_ai_move(black_agent, board)
                print(f"Black AI moves: {move}")
        
        board.push_uci(move)
        env.board = board  # Update the environment's board
    
    print_board(board)
    print(f"Game Over. Result: {board.result()}")

if __name__ == "__main__":
    white_model = load_model('white_model_100000.pth')
    black_model = load_model('black_model_100000.pth')
    
    white_agent = ChessAgent(chess.WHITE)
    white_agent.model = white_model
    
    black_agent = ChessAgent(chess.BLACK)
    black_agent.model = black_model
    
    play_game(white_agent, black_agent)