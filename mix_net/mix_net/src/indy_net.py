import torch
import torch.nn as nn


def outputActivation(x):
    """Custom activation for output layer (Graves, 2015)

    Arguments:
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class IndyNet(nn.Module):

    # Initialization
    def __init__(self, args):
        super(IndyNet, self).__init__()

        # Unpack arguments
        self.args = args

        # Use gpu flag
        self.use_cuda = args["use_cuda"]

        # Sizes of network layers
        self.encoder_size = args["encoder_size"]
        self.decoder_size = args["decoder_size"]
        self.out_length = args["out_length"]
        self.dyn_embedding_size = args["dyn_embedding_size"]
        self.input_embedding_size = args["input_embedding_size"]
        self.enc_dec_layer = args["enc_dec_layer"]
        self.ego_awareness = args["ego_awareness"]

        # Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        if "lstm" in self.enc_dec_layer:
            rnn_encoding = torch.nn.LSTM
        elif "gru" in self.enc_dec_layer:
            rnn_encoding = torch.nn.GRU

        # Encoder LSTM
        # Hist
        self.enc_lstm_hist = rnn_encoding(
            self.input_embedding_size, self.encoder_size, 1
        )

        # Boundaries
        self.enc_lstm_left = rnn_encoding(
            self.input_embedding_size, self.encoder_size, 1
        )
        self.enc_lstm_right = rnn_encoding(
            self.input_embedding_size, self.encoder_size, 1
        )

        if self.ego_awareness:
            # Ego
            self.enc_lstm_ego = rnn_encoding(
                self.input_embedding_size, self.encoder_size, 1
            )
            no_encoders = 4
        else:
            no_encoders = 3

        # decoder:
        decoder_types = ["original", "iterative_hidden", "iterative"]
        assert (
            args["decoder_type"] in decoder_types
        ), "expected decoder_type to be one of {}. Received: {}".format(
            decoder_types, args["decoder_type"]
        )

        if args["decoder_type"] == "original":
            # Encoder hidden state embedder:
            self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

            # Decoder LSTM
            self.dec_lstm = rnn_encoding(
                no_encoders * self.dyn_embedding_size, self.decoder_size
            )
        elif args["decoder_type"] == "iterative_hidden":
            # Embedder to create the cell and hidden state of the decoder from the ones of the encoder:
            self._c_embedder = torch.nn.Linear(3 * self.encoder_size, self.decoder_size)
            self._h_embedder = torch.nn.Linear(3 * self.encoder_size, self.decoder_size)

            # Decoder LSTM:
            self.dec_lstm = rnn_encoding(2, self.decoder_size)
        else:
            pass

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 5)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # migrating the model parameters to the chosen device:
        if args["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA as device for IndyNet")
        else:
            self.device = torch.device("cpu")
            print("Using CPU as device for IndyNet")

        self.to(self.device)

    # Forward Pass
    def forward(self, hist, left_bound, right_bound, ego=None):

        if self.args["decoder_type"] == "original":
            # Forward pass hist:
            hist_enc = self.get_hidden_state(
                self.enc_lstm_hist(self.leaky_relu(self.ip_emb(hist))),
                self.enc_dec_layer,
            )
            # Forward pass left_bound:
            left_enc = self.get_hidden_state(
                self.enc_lstm_left(self.leaky_relu(self.ip_emb(left_bound))),
                self.enc_dec_layer,
            )

            # Forward pass left_bound:
            right_enc = self.get_hidden_state(
                self.enc_lstm_right(self.leaky_relu(self.ip_emb(right_bound))),
                self.enc_dec_layer,
            )

            if self.ego_awareness:
                # Forward pass ego:
                ego_enc = self.get_hidden_state(
                    self.enc_lstm_ego(self.leaky_relu(self.ip_emb(ego))),
                    self.enc_dec_layer,
                )

            hist_enc = self.leaky_relu(
                self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2]))
            )
            left_enc = self.leaky_relu(
                self.dyn_emb(left_enc.view(left_enc.shape[1], left_enc.shape[2]))
            )
            right_enc = self.leaky_relu(
                self.dyn_emb(right_enc.view(right_enc.shape[1], right_enc.shape[2]))
            )

            if self.ego_awareness:
                ego_enc = self.leaky_relu(
                    self.dyn_emb(ego_enc.view(ego_enc.shape[1], ego_enc.shape[2]))
                )

            # Concatenate encodings:
            if self.ego_awareness:
                enc = torch.cat((left_enc, hist_enc, ego_enc, right_enc), 1)
            else:
                enc = torch.cat((left_enc, hist_enc, right_enc), 1)
        else:
            _, (hist_h, hist_c) = self.enc_lstm_hist(self.leaky_relu(self.ip_emb(hist)))
            _, (left_h, left_c) = self.enc_lstm_left(
                self.leaky_relu(self.ip_emb(left_bound))
            )
            _, (right_h, right_c) = self.enc_lstm_right(
                self.leaky_relu(self.ip_emb(right_bound))
            )

            enc = {}

            # the initial cell and hidden state of the decoder:
            enc["c"] = torch.tanh(
                self._c_embedder(torch.cat((hist_c, left_c, right_c), -1))
            )
            enc["h"] = torch.tanh(
                self._h_embedder(torch.cat((hist_h, left_h, right_h), -1))
            )

        fut_pred = self.decode(enc)
        return fut_pred

    def decode(self, enc):
        if self.args["decoder_type"] == "original":
            enc = enc.repeat(self.out_length, 1, 1)
            h_dec, _ = self.dec_lstm(enc)
            h_dec = h_dec.permute(1, 0, 2)  # (batch_size, pred_len, decoder_size)
            fut_pred = self.op(h_dec)  # (batch_size, pred_len, out_size)
            fut_pred = fut_pred.permute(1, 0, 2)  # (pred_len, batch_size, out_size)

        else:
            batch_size = enc["h"].shape[1]
            device = enc["h"].device

            _, (h, _) = self.dec_lstm(
                torch.zeros((1, batch_size, 2), device=device), (enc["h"], enc["c"])
            )
            fut_pred = self.op(h)

            for _ in range(1, self.out_length):
                _, (h, _) = self.dec_lstm(torch.unsqueeze(fut_pred[-1, :, :2], dim=0))
                fut_pred = torch.cat((fut_pred, self.op(h)), dim=0)

        fut_pred = outputActivation(fut_pred)

        return fut_pred

    def get_hidden_state(self, layer_function, enc_dec_layer="lstm"):

        if "lstm" in enc_dec_layer:
            _, (hidden_state, _) = layer_function
        elif "gru" in enc_dec_layer:
            _, hidden_state = layer_function

        return hidden_state

    def load_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Successfully loaded model weights from {}".format(weights_path))
