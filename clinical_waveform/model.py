import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.rnn1 = nn.LSTM(n_features, 2 * n_hidden, batch_first=True)
        self.rnn2 = nn.LSTM(2 * n_hidden, n_hidden, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn1(x)
        _, (h, _) = self.rnn2(x)
        return h.squeeze(0)


class Decoder(nn.Module):
    def __init__(self, seq_len, n_hidden):
        super().__init__()
        self.seq_len = seq_len
        self.rnn1 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.rnn2 = nn.LSTM(n_hidden, 2 * n_hidden, batch_first=True)
        self.output_layer = nn.Linear(2 * n_hidden, 1)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, n_hidden):
        super().__init__()
        self.encoder = Encoder(n_features, n_hidden)
        self.decoder = Decoder(seq_len, n_hidden)

    def forward(self, x):
        return self.decoder(self.encoder(x))
