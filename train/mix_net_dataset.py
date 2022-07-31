"""Create dataset for MixNet."""
# Thrid party imports
import torch
from torch.utils.data import Dataset
import numpy as np


class MixNetDataset(Dataset):
    """Dataset for containing data that is needed for training the MixNet."""

    def __init__(self, data, cut_probability, min_len, random_seed=0):
        """Initializes an MixNetDataset from the given data.

        args:
            data: (dict), already loaded data with the correct keys:
                "hist": (list of 2D lists) the history trajectories.
                "fut": (list of 2D lists) the groundtruth future trajectories.
                "fut_inds": (list of lists) the indices of the nearest centerline points
                    corresponding to the ground truth prediction.
                "left_bd": (list of 2D lists) left track boundary snippet.
                "right_bd": (list of 2D lists) right track boundary snippet.
        """

        self._cut_probability = cut_probability
        self._min_len = min_len
        self._rng = np.random.default_rng(
            random_seed
        )  # random number generator with seed

        # checking the data:
        keys_gotten = list(data.keys())
        keys_needed = ["hist", "fut", "fut_inds", "left_bd", "right_bd"]

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

        hist = self.D["hist"][
            idx, ::-1, :
        ].copy()  # Changing the order of history back to normal
        fut = self.D["fut"][idx, :, :]
        fut_inds = self.D["fut_inds"][idx]
        left_boundary = self.D["left_bd"][idx, :, :]
        right_boundary = self.D["right_bd"][idx, :, :]

        return hist, fut, fut_inds, left_boundary, right_boundary

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
        len_in = samples[0][0].shape[0]
        len_out = samples[0][1].shape[0]
        len_bound = samples[0][3].shape[0]

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros((batch_size, len_in, 2), dtype=torch.float32)
        fut_batch = torch.zeros((batch_size, len_out, 2), dtype=torch.float32)
        fut_inds_batch = torch.zeros((batch_size, len_out), dtype=torch.int16)
        left_bd_batch = torch.zeros((batch_size, len_bound, 2), dtype=torch.float32)
        right_bd_batch = torch.zeros((batch_size, len_bound, 2), dtype=torch.float32)

        for i, (hist, fut, fut_inds, left_bd, right_bd) in enumerate(samples):
            # changing the length with a given probability:
            if self._rng.binomial(size=1, n=1, p=self._cut_probability):
                hist_len = int(self._rng.uniform(self._min_len, hist.shape[0]))
            else:
                hist_len = hist.shape[0]

            # Filling up the tensors, and also CHANGING THE ORDER OF HISTORY!!!:
            hist_batch[i, :hist_len, :] = torch.from_numpy(hist[-hist_len:, :])
            fut_batch[i, :, :] = torch.from_numpy(fut)
            fut_inds_batch[i, :] = torch.from_numpy(fut_inds)
            left_bd_batch[i, :, :] = torch.from_numpy(left_bd)
            right_bd_batch[i, :, :] = torch.from_numpy(right_bd)

        return hist_batch, fut_batch, fut_inds_batch, left_bd_batch, right_bd_batch
