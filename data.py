"""Data class: new datasets are added here."""
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

logging.getLogger(__name__).addHandler(logging.NullHandler())


class DataSpace:
    """A dataset (real or synthetic) that can be sampled from.

    data: can be images, in which case we need to generate representations
          or pre-calculated representations (glove, dinovit, etc).
    labels: item class labels
    Datasets also need test split (data, labels).

    These are synthetic datsets, or very simple image datasets, represented by
    UMAP or an autoencoder: not good enough.
    """

    def __init__(self):
        """Set up train, test splits."""
        self.data, self.labels = self.make_data()

        self.test_data, self.test_labels = self.make_test_data()

        if len(self.data.shape) > 2:
            self.data_dim = np.multiply(*self.data.shape[1:])
        else:  # eg synth data is already flat
            self.data_dim = self.data.shape[1]

    def setup_rep_scaler(self, reps) -> None:
        """Create representation scaler/normalizer."""
        # this scaler is scaled to *all* the data, not just single learner's.
        # NB needs to fit representations, not raw data.
        self.scaler = StandardScaler()
        self.scaler.fit(reps)

    def make_data(self):
        raise NotImplementedError

    def make_test_data(self):
        raise NotImplementedError

    def sample_points(self, N, rng):
        """Returns (original_data, representations, label) for N sampled points."""
        idx = rng.choice(range(len(self.labels)), size=N, replace=False)
        sample_label = self.labels[idx]
        sample_reps = self.representations[idx,:]  # scaled already
        sample_data = self.data[idx,:]

        return sample_data, sample_reps, sample_label

    def create_representations(self):
        raise NotImplementedError

    def setup_representations(self, rep_dim, rep_type, *kwargs):
        raise NotImplementedError
