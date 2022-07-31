"""Create IndyNet Dataset."""
# Thrid party imports
import torch
from torch.utils.data import Dataset
import numpy as np


# Dataset class for Indy
class IndyDataset(Dataset):
    """Dataset for containing data that is needed for training the IndyNet."""

    def __init__(self, data, cut_probability, min_len, random_seed=0):
        """Initializes an IndyDataset from the given data.

        args:
            data: (dict), already loaded data with the correct keys:
                "id": (list) the IDs
                "hist": (list of 2D lists) the history trajectories.
                "fut": (list of 2D lists) the groundtruth future trajectories.
                "left_bd": (list of 2D lists) left track boundary snippet.
                "right_bd": (list of 2D lists) right track boundary snippet.
            cut_probability: (float in [0, 1]), the probability that a datasample is going
                to be cut in the collate function to be able to train with different history lengths.
            min_len: (float), the minimum length of a history to which it can be cut to.
            random_seed: (float, default=0), the random seed with which the random lengths are generated.
        """

        self._cut_probability = cut_probability
        self._min_len = min_len
        self._rng = np.random.default_rng(
            random_seed
        )  # random number generator with seed

        # checking the data:
        keys_gotten = list(data.keys())
        keys_needed = ["id", "hist", "fut", "left_bd", "right_bd"]

        for key in keys_needed:
            assert (
                key in keys_gotten
            ), "Key {} is not found in the given data. Keys found: {}".format(
                key, keys_gotten
            )

        self.D = data

    def __len__(self):
        """Must be defined for a Dataset"""

        return self.D["hist"].shape[0]

    def __getitem__(self, idx):
        """must be defined for a Dataset"""

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        smpl_id = self.D["id"][idx]
        hist = self.D["hist"][
            idx, ::-1, :
        ].copy()  # Changing the order of the history back to normal!!!!
        fut = self.D["fut"][idx, :, :]
        left_boundary = self.D["left_bd"][idx, :, :]
        right_boundary = self.D["right_bd"][idx, :, :]
        try:
            ego = self.D["ego"][idx]
        except Exception:
            ego = None

        return smpl_id, hist, fut, left_boundary, right_boundary, ego

    def collate_fn(self, samples):
        """Function that defines how the samples are collated when using mini-batch training.

        This is needed, because although every datasample has the same length originally, we would
        like to train with different history lengths. For every history, we decide:
            * With a probability it can keep its original length (30 usually)
            * If the sample does not keep its original lenght, its length is chosen randomly
                between a min and max bound.
        The collate function is needed in order to collate these different length histories.
        The remaining place is filled up with zeros.
        """

        batch_size = len(samples)
        len_in = samples[0][1].shape[0]
        len_out = samples[0][2].shape[0]
        len_bound = samples[0][3].shape[0]

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros((len_in, batch_size, 2))
        ego_batch = torch.zeros((len_in, batch_size, 2))
        fut_batch = torch.zeros((len_out, batch_size, 2))
        left_bd_batch = torch.zeros((len_bound, batch_size, 2))
        right_bd_batch = torch.zeros((len_bound, batch_size, 2))

        smpl_ids = []
        for sampleId, (smpl_id, hist, fut, left_bd, right_bd, ego) in enumerate(
            samples
        ):

            # changing the length with a given probability:
            if self._rng.binomial(size=1, n=1, p=self._cut_probability):
                hist_len = int(self._rng.uniform(self._min_len, hist.shape[0]))
            else:
                hist_len = hist.shape[0]

            # Filling up the tensors, and also CHANGING THE ORDER OF HISTORY!!!:
            # After chaning the order hist[0] is the oldest history and hist[-1] ist the latest position.
            hist_batch[:hist_len, sampleId, :] = torch.from_numpy(hist[-hist_len:, :])
            fut_batch[:, sampleId, :] = torch.from_numpy(fut)
            left_bd_batch[:, sampleId, :] = torch.from_numpy(left_bd)
            right_bd_batch[:, sampleId, :] = torch.from_numpy(right_bd)

            try:
                ego_batch[0 : len(ego), sampleId, 0] = torch.from_numpy(ego[:, 0])
                ego_batch[0 : len(ego), sampleId, 1] = torch.from_numpy(ego[:, 1])
            except Exception:
                ego_batch = None

            smpl_ids.append(smpl_id)

        return smpl_ids, hist_batch, fut_batch, left_bd_batch, right_bd_batch, ego_batch
