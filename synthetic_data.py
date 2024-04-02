"""Data class: new datasets are added here."""
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from data import DataSpace

logging.getLogger(__name__).addHandler(logging.NullHandler())

class SyntheticData(DataSpace):
    """Generate data from Gaussian distributions.

    Here, data and representations are equivalent.
    Data dimensionality equals representational dimensionality.
    """

    def __init__(self,
                 num_dimensions,
                 num_groups,
                 cluster_std=1.0,
                 centers=None,
                 rng=None,
                 ):

        self.rep_dim = num_dimensions
        self.num_dimensions = num_dimensions
        self.num_groups = num_groups
        self.cluster_std = cluster_std
        self.rng = rng
        self.centers = None   # initialised by make_data
        self.dataset_str = "synth"
        super().__init__()  # calls make_data


    def make_data(self, num_points:int =1000000) -> tuple:
        """Create many many points, from which generations then sample."""
        logging.info((f"Synthetic data: D={self.num_dimensions}",
                      f"C={self.num_groups}, cluster_std={self.cluster_std}"))

        centers = self.centers
        if self.centers is None:
            centers  = self.num_groups
        else:
            assert len(centers) == self.num_groups

        data, labels, centers = make_blobs(
            n_samples=num_points,
            n_features=self.num_dimensions,
            centers=centers,  # num_groups or actual centers
            cluster_std=self.cluster_std,
            random_state=self.rng,
            return_centers=True,
        )
        logging.info(f"Synth cluster centers {centers}")

        self.centers = centers

        return data, labels

    def make_test_data(self, num_points:int=100) -> tuple:
        """Test points are made from the same centers as make_data.

        Make 1/10th as much data as training
        """
        data, labels, centers = make_blobs(
            n_samples=num_points,
            n_features=self.num_dimensions,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=self.rng,
            return_centers=True,
        )
        return data, labels


    def create_representations(self):
        return self.data  # no-op, data == representations

    def setup_representations(self, rep_dim, rep_type, *kwargs):
        """Return the synth data 'representations', scaled by self.scaler.

        With synth data, never use cached representations: overriding method here.
        """
        representations = self.create_representations()
        self.setup_rep_scaler(representations)
        self.representations = self.scaler.transform(representations)
