import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.value_head = nn.Linear(512, 1)
        self.policy_head = nn.Linear(512, 64 * 64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)  # Reshape input to 12 channels (6 piece types * 2 colors)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = torch.tanh(self.value_head(x))
        policy = F.log_softmax(self.policy_head(x), dim=1)
        return value, policy