import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model import ChessNet
import chess
from collections import deque
import random

class ChessAgent:
    def __init__(self, color, initial_epsilon=0.9, epsilon_decay=0.9999, min_epsilon=0.01, lr=0.001):
        self.board = chess.Board()
        self.color = 'white' if color == chess.WHITE else 'black'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.model_file = self.generate_model_filename(self.color)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.memory = deque(maxlen=100000)
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.illegal_moves = {}
        self.illegal_moves_counter = 0
        self.recent_rewards = deque(maxlen=100)

    def load_model(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    def save_model(self):
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, self.model_file)
        torch.save(self.model.state_dict(), filename)

    def initialize_model_file(self, color):
        if self.model_file is None:
            self.model_file = self.generate_model_filename(color)

    def generate_model_filename(self, color):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{color.lower()}_model_{timestamp}.pth"

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)

    def clear_old_illegal_moves(self, max_size=10000):
        if len(self.illegal_moves) > max_size:
            # Remove oldest entries
            sorted_moves = sorted(self.illegal_moves.items(), key=lambda x: len(x[1]), reverse=True)
            self.illegal_moves = dict(sorted_moves[:max_size])

    def update_illegal_move(self, state, action):
        state_key = self.state_to_key(state)
        if state_key not in self.illegal_moves:
            self.illegal_moves[state_key] = set()
        self.illegal_moves[state_key].add(action)
        self.illegal_moves_counter += 1

        # Call clear_old_illegal_moves every 1000 illegal moves
        if self.illegal_moves_counter % 1000 == 0:
            self.clear_old_illegal_moves()

    def select_action(self, state, legal_moves):
        # Check for obvious captures or promotions
        for move in legal_moves:
            chess_move = chess.Move.from_uci(move)
            if self.board.is_capture(chess_move) or chess_move.promotion:
                return move

        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            value, policy = self.model(state_tensor)
            self.model.train()
        
        policy = policy.squeeze().cpu().numpy()
        legal_move_indices = [self.move_to_index(move) for move in legal_moves]
        
        legal_move_probs = policy[legal_move_indices]
        
        # Ensure probabilities are non-negative
        legal_move_probs = np.maximum(legal_move_probs, 0)
        
        if np.sum(legal_move_probs) == 0:
            return random.choice(legal_moves)

        # Apply progressive widening
        n_consider = max(1, int(len(legal_moves) * (1 - self.epsilon)))
        top_moves = sorted(zip(legal_moves, legal_move_probs), key=lambda x: x[1], reverse=True)[:n_consider]
        
        # Ensure we have positive probabilities
        top_move_probs = [prob for _, prob in top_moves]
        if sum(top_move_probs) == 0:
            return random.choice(legal_moves)
        
        chosen_move, _ = random.choices([move for move, _ in top_moves], weights=top_move_probs, k=1)[0]
        return chosen_move

    def update_board(self, board):
        self.board = board

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0

        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([self.move_to_index(a) for a in action]).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        value, policy = self.model(state)
        next_value, _ = self.model(next_state)

        td_target = reward + (1 - done) * 0.99 * next_value.squeeze(1)
        value_loss = F.mse_loss(value.squeeze(1), td_target.detach())

        policy_loss = -torch.mean(torch.sum(policy * F.one_hot(action, 64*64), dim=1) * (td_target - value.squeeze(1)).detach())

        illegal_move_loss = 0
        for i, s in enumerate(state):
            state_key = self.state_to_key(s.cpu().numpy())
            if state_key in self.illegal_moves:
                for illegal_move in self.illegal_moves[state_key]:
                    illegal_index = self.move_to_index(illegal_move)
                    illegal_move_loss += policy[i, illegal_index]

        loss = value_loss + policy_loss + 0.1 * illegal_move_loss

        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()

        return grad_norm.item()  # Return the gradient norm

    def state_to_key(self, state):
        return tuple(state)
    
    def move_to_index(self, move):
        from_square = chess.SQUARE_NAMES.index(move[:2])
        to_square = chess.SQUARE_NAMES.index(move[2:4])
        return from_square * 64 + to_square

    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]