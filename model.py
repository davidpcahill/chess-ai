import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 8 * 8 + 1, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        
        self.value_head = nn.Linear(512, 1)
        self.policy_head = nn.Linear(512, 64 * 64)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        player = x[:, -1].unsqueeze(1)
        x = x[:, :-1].view(-1, 12, 8, 8)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) + x
        x = F.relu(self.bn3(self.conv3(x))) + x
        
        x = x.view(x.size(0), -1)
        x = torch.cat([x, player], dim=1)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        
        value = torch.tanh(self.value_head(x))
        policy = F.log_softmax(self.policy_head(x), dim=1)
        
        return value, policy