import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_output):
        # rnn_output: (batch_size, seq_len, hidden_dim)
        weights = torch.softmax(self.attention(rnn_output), dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context

class BiLSTMAttentionSignClassifier(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=128, num_layers=2, num_classes=37, dropout=0.3):
        super(BiLSTMAttentionSignClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out
