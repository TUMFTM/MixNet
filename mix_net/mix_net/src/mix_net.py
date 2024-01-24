import json
import torch
import torch.nn as nn
from collections import OrderedDict


class MixNet(nn.Module):
    """Neural Network to predict the future trajectory of a vehicle based on its history.
    It predicts mixing weights for mixing the boundaries, the centerline and the raceline.
    Also, it predicts section wise constant accelerations and an initial velocity to
    compute the velocity profile from.
    """

    def __init__(self, params):
        """Initializes a MixNet object."""
        super(MixNet, self).__init__()

        self._params = params

        # Input embedding layer:
        self._ip_emb = torch.nn.Linear(2, params["encoder"]["in_size"])

        # History encoder LSTM:
        self._enc_hist = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Boundary encoders:
        self._enc_left_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        self._enc_right_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Linear stack that outputs the path mixture ratios:
        self._mix_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"] * 3,
            hidden_sizes=params["mixer_linear_stack"]["hidden_sizes"],
            out_size=params["mixer_linear_stack"]["out_size"],
            name="mix",
        )

        # Linear stack for outputting the initial velocity:
        self._vel_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"],
            hidden_sizes=params["init_vel_linear_stack"]["hidden_sizes"],
            out_size=params["init_vel_linear_stack"]["out_size"],
            name="vel",
        )

        # dynamic embedder between the encoder and the decoder:
        self._dyn_embedder = nn.Linear(
            params["encoder"]["hidden_size"] * 3, params["acc_decoder"]["in_size"]
        )

        # acceleration decoder:
        self._acc_decoder = nn.LSTM(
            params["acc_decoder"]["in_size"],
            params["acc_decoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # output linear layer of the acceleration decoder:
        self._acc_out_layer = nn.Linear(params["acc_decoder"]["hidden_size"], 1)

        # migrating the model parameters to the chosen device:
        if params["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA as device for MixNet")
        else:
            self.device = torch.device("cpu")
            print("Using CPU as device for MixNet")

        self.to(self.device)

    def forward(self, hist, left_bound, right_bound):
        """Implements the forward pass of the model.

        args:
            hist: [tensor with shape=(batch_size, hist_len, 2)]
            left_bound: [tensor with shape=(batch_size, boundary_len, 2)]
            right_bound: [tensor with shape=(batch_size, boundary_len, 2)]

        returns:
            mix_out: [tensor with shape=(batch_size, out_size)]: The path mixing ratios in the order:
                left_ratio, right_ratio, center_ratio, race_ratio
            vel_out: [tensor with shape=(batch_size, 1)]: The initial velocity of the velocity profile
            acc_out: [tensor with shape=(batch_size, num_acc_sections)]: The accelerations in the sections
        """

        # encoders:
        _, (hist_h, _) = self._enc_hist(self._ip_emb(hist.to(self.device)))
        _, (left_h, _) = self._enc_left_bound(self._ip_emb(left_bound.to(self.device)))
        _, (right_h, _) = self._enc_right_bound(
            self._ip_emb(right_bound.to(self.device))
        )

        # concatenate and squeeze encodings:
        enc = torch.squeeze(torch.cat((hist_h, left_h, right_h), 2), dim=0)

        # path mixture through softmax:
        mix_out = torch.softmax(self._mix_out_layers(enc), dim=1)

        # initial velocity:
        vel_out = self._vel_out_layers(torch.squeeze(hist_h, dim=0))
        vel_out = torch.sigmoid(vel_out)
        vel_out = vel_out * self._params["init_vel_linear_stack"]["max_vel"]

        # acceleration decoding:
        dec_input = torch.relu(self._dyn_embedder(enc)).unsqueeze(dim=1)
        dec_input = dec_input.repeat(
            1, self._params["acc_decoder"]["num_acc_sections"], 1
        )
        acc_out, _ = self._acc_decoder(dec_input)
        acc_out = torch.squeeze(self._acc_out_layer(torch.relu(acc_out)), dim=2)
        acc_out = torch.tanh(acc_out) * self._params["acc_decoder"]["max_acc"]

        return mix_out, vel_out, acc_out

    def load_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Successfully loaded model weights from {}".format(weights_path))

    def _get_linear_stack(
        self, in_size: int, hidden_sizes: list, out_size: int, name: str
    ):
        """Creates a stack of linear layers with the given sizes and with relu activation."""

        layer_sizes = []
        layer_sizes.append(in_size)  # The input size of the linear stack
        layer_sizes += hidden_sizes  # The hidden layer sizes specified in params
        layer_sizes.append(out_size)  # The output size specified in the params

        layer_list = []
        for i in range(len(layer_sizes) - 1):
            layer_name = name + "linear" + str(i + 1)
            act_name = name + "relu" + str(i + 1)
            layer_list.append(
                (layer_name, nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            )
            layer_list.append((act_name, nn.ReLU()))

        # removing the last ReLU layer:
        layer_list = layer_list[:-1]

        return nn.Sequential(OrderedDict(layer_list))

    def get_params(self):
        """Accessor for the params of the network."""
        return self._params


if __name__ == "__main__":
    param_file = "mod_prediction/utils/mix_net/params/net_params.json"
    with open(param_file, "r") as fp:
        params = json.load(fp)

    batch_size = 32
    hist_len = 30
    bound_len = 50

    # random inputs:
    hist = torch.rand((batch_size, hist_len, 2))
    left_bound = torch.rand((batch_size, bound_len, 2))
    right_bound = torch.rand((batch_size, bound_len, 2))

    net = MixNet(params)

    mix_out, vel_out, acc_out = net(hist, left_bound, right_bound)

    # printing the output shapes as a sanity check:
    print(
        "output shape: {}, should be: {}".format(
            mix_out.shape, (batch_size, params["mixer_linear_stack"]["out_size"])
        )
    )

    print("output shape: {}, should be: {}".format(vel_out.shape, (batch_size, 1)))

    print(
        "output shape: {}, should be: {}".format(
            acc_out.shape, (batch_size, params["acc_decoder"]["num_acc_sections"])
        )
    )

    print(vel_out)
