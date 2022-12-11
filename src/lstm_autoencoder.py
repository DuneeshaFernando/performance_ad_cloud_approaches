import torch
import torch.nn as nn
import src.trainer_utils as utils

device = utils.get_default_device()

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=2, num_neurons=[]):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder_layers = nn.ModuleList()
        for layer_n in range(num_layers, 0, -1):
            h_layer = nn.LSTM(
                input_size=num_neurons[layer_n],
                hidden_size=num_neurons[layer_n - 1],
                num_layers=1,
                batch_first=True
            )
            self.encoder_layers.append(h_layer)

    def forward(self, x):
        for encoder_layer in self.encoder_layers[:-1]:
            x, _ = encoder_layer(x)
        x, (hidden_n, _) = self.encoder_layers[-1](x)
        temp = hidden_n.reshape((-1, 1, self.embedding_dim))
        return temp


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, n_features=1, num_layers=2, num_neurons=[]):
        super(Decoder, self).__init__()
        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.n_features = n_features

        self.decoder_layers = nn.ModuleList()
        for layer_n in range(num_layers):
            if layer_n == 0:
                h_layer = nn.LSTM(
                        input_size=num_neurons[layer_n],
                        hidden_size=num_neurons[layer_n],
                        num_layers=1,
                        batch_first=True
                )
            else:
                h_layer = nn.LSTM(
                    input_size=num_neurons[layer_n - 1],
                    hidden_size=num_neurons[layer_n],
                    num_layers=1,
                    batch_first=True
                )
            self.decoder_layers.append(h_layer)

        self.decoder_layers.append(nn.Linear(num_neurons[num_layers - 1], num_neurons[num_layers]))

    def forward(self, x):
        x = x.repeat((1, self.seq_len, 1))
        for decoder_layer in self.decoder_layers[:-1]:
            x, _ = decoder_layer(x)
        x = self.decoder_layers[-1](x)
        return x


class LstmAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=2):
        super(LstmAutoencoder, self).__init__()

        num_neurons = []
        num_neurons.append(embedding_dim)
        hidden_dim = embedding_dim
        for l in range(num_layers - 1):
            hidden_dim = int(hidden_dim * 2)
            num_neurons.append(hidden_dim)
        num_neurons.append(n_features)

        self.encoder = Encoder(seq_len, n_features, embedding_dim, num_layers, num_neurons).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features, num_layers, num_neurons).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def training(epochs, lstm_autoencoder_model, train_loader, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_autoencoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = utils.to_device(batch, device)
            recon = lstm_autoencoder_model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss: {loss.item():.4f}')


def testing(lstm_autoencoder_model, test_loader):
    results = []
    for [batch] in test_loader:
        batch = utils.to_device(batch, device)
        with torch.no_grad():
            recon = lstm_autoencoder_model(batch)
        results.append(torch.mean((batch - recon) ** 2, axis=(1,2)))
    return results
