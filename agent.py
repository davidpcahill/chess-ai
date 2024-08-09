import torch
import torch.optim as optim
import numpy as np
from model import ChessNet
import chess
from collections import deque
import random

class ChessAgent:
    def __init__(self, color):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.color = color
        self.memory = deque(maxlen=100000)

    def select_action(self, state, legal_moves, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(legal_moves)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, policy = self.model(state)
        
        legal_move_mask = torch.zeros(64 * 64).to(self.device)
        for move in legal_moves:
            legal_move_mask[self.move_to_index(move)] = 1
        masked_policy = policy + torch.log(legal_move_mask)

        move_index = torch.argmax(masked_policy).item()
        return self.index_to_move(move_index)

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