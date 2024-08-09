import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import ChessNet
import chess
from collections import deque
import random

class ChessAgent:
    def __init__(self, color, initial_epsilon=0.9, epsilon_decay=0.99995, min_epsilon=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.model_file = self.generate_model_filename(color)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.color = color
        self.memory = deque(maxlen=100000)
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model_file = os.path.basename(model_path)

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
        return f"{color}_model_{timestamp}.pth"

    def select_action(self, state, legal_moves):
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, policy = self.model(state)
        
        policy = policy.squeeze().cpu().numpy()
        legal_move_indices = [self.move_to_index(move) for move in legal_moves]
        legal_move_probs = policy[legal_move_indices]
        
        if np.sum(legal_move_probs) == 0:
            return random.choice(legal_moves)
        
        probs = legal_move_probs / np.sum(legal_move_probs)
        chosen_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[chosen_idx]

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

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

        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move_to_index(self, move):
        from_square = chess.SQUARE_NAMES.index(move[:2])
        to_square = chess.SQUARE_NAMES.index(move[2:4])
        return from_square * 64 + to_square

    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]