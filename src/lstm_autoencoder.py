import torch
import torch.nn as nn
import src.trainer_utils as utils

device = utils.get_default_device()

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((-1, 1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.hidden_dim, self.n_features = 2 * embedding_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat((1, self.seq_len, 1))
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.output_layer(x)


class LstmAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LstmAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def training(epochs, lstm_autoencoder_model, train_loader):
    history = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_autoencoder_model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = utils.to_device(batch, device)
            recon = lstm_autoencoder_model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f'Epoch:{epoch + 1}, Loss: {loss.item():.4f}')
        history.append((epoch, batch, recon))
    return history


def testing(lstm_autoencoder_model, test_loader):
    results = []
    for [batch] in test_loader:
        batch = utils.to_device(batch, device)
        with torch.no_grad():
            recon = lstm_autoencoder_model(batch)
        results.append(torch.mean((batch - recon) ** 2, axis=(1,2)))
    return results
