"""Base Codebook class."""
import logging

import numpy as np
import numpy.random as nprand
from numpy import ma  # masked arrays
from scipy.spatial import distance
from sklearn import metrics, svm
from sklearn.feature_selection import VarianceThreshold

SEED = 26


class Codebook:
    def __init__(
        self,
        data,
        labels,
        num_classifiers: int,
        sample_M: int,
        classifier_accuracy_threshold: float = 0.8,
        teacher_labels=None,
        teacher_codewords=None,
        svm_C=0.1,
        rng=None,
    ):
        self.data_reps = data  # scaled in dataspace

        # SVM classifier to use when setting up codebook columns
        self.svm_C = svm_C  # default sklearn C is 0.1 not 1 ; high regularisation
        self.clf_f = lambda: svm.SVC(
            kernel="linear",
            C=self.svm_C,
        )

        self.labels = labels
        self.encoder_D = self.data_reps.shape[-1]
        self.len_data = self.data_reps.shape[0]

        self.data_to_rows_map = np.arange(0, self.len_data, 1)
        # self.rows = defaultdict(list)
        self.rows_data_idx = {i: [i] for i in range(self.len_data)}

        if rng is None:
            logging.info(f"Codebook: setting rng with seed {SEED}")
            self.rng = nprand.default_rng(SEED)
        else:
            self.rng = rng

        self.sample_M = sample_M
        self.classifiers = []  # column index -> classifier
        self.classifiers_in_use = []  # index into classifiers
        self.classifier_accuracy_threshold = classifier_accuracy_threshold

        self.codebook = self.make_codebook(
            num_classifiers,
            teacher_labels,
            teacher_codewords,
        )

    def make_codebook(self, D: int, teacher_labels=None, teacher_codewords=None):
        """Set up D classifiers, using either teacher labels/codewords or random init.

        NB: teacher might have much shorter codewords than D, due to column
        filtering and merging. So add any new columns as random classifiers.
        """
        code_columns = []

        row_reps = self.get_row_reps()

        while len(self.classifiers) < D:
            if teacher_labels is not None:  # learn a classifier for a random label
                col, d, accuracy = self.setup_classifier_teaching(
                    row_reps,
                    teacher_labels,
                )
            elif (  # learn classifier for codeword feature ci
                teacher_codewords is not None
                and len(self.classifiers) < teacher_codewords.shape[1]
            ):
                ci = len(self.classifiers)  # current classifier
                col, d, accuracy = self.setup_classifier_codeword(
                    row_reps,
                    teacher_codewords[:, ci],
                    index=ci,
                )

            else:  # random classifiers, e.g. at beginning.
                col, d, accuracy = self.setup_classifier_random(row_reps, self.sample_M)
                logging.info(
                    f"Adding a new random classifier {len(self.classifiers)} acc:{accuracy}",
                )

            if accuracy > self.classifier_accuracy_threshold or (
                teacher_codewords is not None
            ):
                self.classifiers_in_use.append(len(self.classifiers))
                self.classifiers.append(d)
                code_columns.append(col)
                logging.debug(
                    f"Classifier {len(self.classifiers)} with acc={accuracy}  predictions/col ones {sum(col)}/{len(col)} predictions: {col[0:20]}",
                )
            else:
                logging.debug(
                    f"BAD Classifier {len(self.classifiers)} with acc={accuracy}  predictions/col ones {sum(col)}/{len(col)}",
                )
        codebook = np.array(code_columns).transpose()

        codebook = ma.masked_array(codebook, mask=np.zeros_like(codebook))

        logging.debug(
            f"cb num rows {len(codebook)} ({len(np.unique(codebook, axis=0))}) num cols {len(code_columns)} ({len(np.unique(code_columns, axis=0))})",
        )
        return codebook

    def get_row_reps(self):
        """Cookbook subclasses are defined by how they represent rows."""

    def setup_classifier_random(self, reps, int_M: int) -> tuple:
        """Args:
        ----
            reps: representations - datapoints at each row
            int_M: number of items to sample for training the classifier
        """
        N = reps.shape[0]

        # generate subsample: indices
        M_idx = self.rng.choice(range(N), int_M, replace=False)
        M = reps[M_idx]

        # generate random labels for subsample
        int_M_pos = self.rng.integers(1, int_M)  # uniform dist over positives
        Y = np.array([1] * int_M_pos + [0] * (int_M - int_M_pos))
        self.rng.shuffle(Y)
        assert len(Y) == int_M

        clf = self.clf_f()
        fit = clf.fit(M, Y)

        preds = clf.predict(reps)

        # training item predictions should be equal to Y - but not always
        preds_M = preds[M_idx]
        training_accuracy = sum([preds_M[j] == Y[j] for j in range(int_M)]) / int_M
        logging.debug(f"D training accuracy {training_accuracy}")

        # pred(iction)s are binary vectors over the number of rows:
        # one prediction is a column in the codebook.
        return preds, clf, training_accuracy

    def setup_classifier_teaching(
        self,
        reps,
        teacher_labels,
        example_class=None,
    ):
        """Args:
        ----
            reps: from self.get_row_reps()
            example class: positive example class  (i.e. using self.labels)
                if none: picked at random.
        """
        N = reps.shape[0]
        Nt = len(teacher_labels)
        # int_M = self.sample_M

        if example_class is None:
            example_class = self.rng.choice(np.unique(teacher_labels))

        in_class = np.flatnonzero(teacher_labels == example_class)
        int_M_pos = len(in_class)
        int_M_neg = int_M_pos  # balance positive and negative class sizes

        # sample a int_M positive/negative examples instead of using all of them
        pos_idx = self.rng.choice(in_class, int_M_pos, replace=False)  # all pos
        neg_idx = self.rng.choice(  # equal number of neg
            np.flatnonzero([self.labels != example_class]),
            int_M_neg,
            replace=False,
        )

        M_idx = np.concatenate([pos_idx, neg_idx])
        assert len(M_idx) == int_M_pos + int_M_neg, f"M_idx shape {M_idx.shape}"
        M = reps[M_idx]

        Y = np.array([1] * int_M_pos + [0] * int_M_neg)

        logging.debug(
            f"D classifier with example class {example_class}, size {int_M_pos} ids {pos_idx[0:10]}",
        )

        clf = self.clf_f()
        fit = clf.fit(M, Y)

        preds = clf.predict(reps)

        # training item predictions should be equal to Y - but not always: check acc
        preds_M = preds[M_idx]
        training_acc = sum([preds_M[j] == Y[j] for j in range(len(M_idx))]) / len(M_idx)

        # pred(iction)s are binary vectors over the number of rows:
        # one prediction is a column in the codebook.
        return preds, clf, training_acc

    def setup_classifier_codeword(self, reps, teacher_column, index=0):
        """Initialize codebook with given teacher codewords *then* learn the classifiers."""
        Nt = len(teacher_column)  # number of rows (from top of data)
        M = reps[:Nt]
        Y = teacher_column

        clf = self.clf_f()

        # if Y is all 0/1, svm won't work - generate 'predictions' that match Y
        # this column will later be removed when removing low variance columns
        if len(np.unique(Y)) == 1:  # only 0/1 in column, bad
            logging.info(f"BAD teacher column {Y[0:10]} at {index}")
            if Y[0] == 1:
                preds = np.ones(reps.shape[0])
            else:
                preds = np.zeros(reps.shape[0])

        else:
            _ = clf.fit(M, Y)
            preds = clf.predict(reps)

        # training item predictions should be equal to Y - but not always
        preds_M = preds[:Nt]
        training_accuracy = sum([preds_M[j] == Y[j] for j in range(Nt)]) / Nt

        return preds, clf, training_accuracy

    def generate_codewords(self, datapoints, coerce_to_codebook=False):
        """Make the codeword for a datapoint, given codebook classifiers."""
        column_predictions = []
        for clf_idx in range(len(self.classifiers)):
            if clf_idx in self.classifiers_in_use:
                clf = self.classifiers[clf_idx]
                col_prediction = clf.predict(datapoints)  # this should be one bit
                column_predictions.append(col_prediction)

        codeword_predictions = np.array(column_predictions, dtype=int).transpose()
        logging.debug(
            f"predictions for new points {codeword_predictions.shape}\n {codeword_predictions}",
        )
        cw_labels = None

        if coerce_to_codebook:
            """Codeword generations have to match the closest (hamming distance) row"""
            coerced_predictions = np.apply_along_axis(  # row-wise
                self.return_best_match,
                1,
                codeword_predictions,
            )
            cdiff = np.count_nonzero(codeword_predictions - coerced_predictions)
            codeword_predictions = coerced_predictions

            # NB these labels do not match anything in self.labels (those are gold)
            cw_labels = np.zeros(datapoints.shape[0], dtype=int)
            for cwi, unique_cw in enumerate(np.unique(codeword_predictions, axis=0)):
                idx = ma.where((codeword_predictions == unique_cw).all(axis=1))[0]
                cw_labels[idx] = cwi
            logging.debug(f"coerced predictions \n {codeword_predictions}")
            logging.debug(f"predicted labels {cw_labels}")
            logging.debug(
                f"coerced predictions: unique labels {len(np.unique(codeword_predictions, axis=0))} of {len(codeword_predictions)}; coersion diff {cdiff}",
            )

        return codeword_predictions, cw_labels

    def return_best_match(self, prediction):
        """Find current row with best/smallest hamming distance to prediction."""
        distances = distance.cdist(
            self.get_unmasked_codebook(),
            np.array([prediction]),
            "hamming",
        )
        best_row = np.argmin(distances)
        return self.get_unmasked_codebook()[best_row]

    def label_identical_rows(self):
        """Rows with identical codewords are given the same (cluster) label.

        For initial clustering, identify which rows are identical & construct a
        corresponding labelling to pass to CB.update_rows(). This will be
        faster than the large pairwise distance calculations when there are
        many identical rows.

        """
        labelling = np.zeros(self.codebook.shape[0], dtype=int)

        for cwi, unique_cw in enumerate(np.unique(self.codebook, axis=0)):
            idx = ma.where((self.codebook == unique_cw).all(axis=1))[0]
            labelling[idx] = cwi

        return labelling

    def evaluate_row_classes(self):
        """Do row clusters correspond to gold clusters/labels?

        NB that labels can be swapped; evaluate (unsupervised) clusters.
        """
        pred_labels = self.data_to_rows_map
        # ari = metrics.adjusted_rand_score(self.labels, pred_labels)
        # ami = metrics.adjusted_mutual_info_score(self.labels, pred_labels)
        homogenity = metrics.homogeneity_score(self.labels, pred_labels)
        completeness = metrics.completeness_score(self.labels, pred_labels)
        vmeasure = metrics.v_measure_score(self.labels, pred_labels)

        scores = {
            # "ari": float(ari),
            # "ami": float(ami),
            "H": float(homogenity),
            "C": float(completeness),
            "VM": float(vmeasure),
        }
        return scores

    def get_unmasked_codebook(self):
        """Return codebook array with current classifier columns."""
        return np.array(self.codebook[:, self.classifiers_in_use])

    def mask_identical_columns(self):
        """Mask/remove columns that are identical in codebook
        """
        columns = self.codebook.transpose()

        if len(np.unique(columns, axis=0)) < len(self.classifiers_in_use):
            logging.info(
                f"Removing IDENTICAL features: {len(np.unique(columns, axis=0))} unique"
                f" of {len(self.classifiers_in_use)}",
            )
        for unique_c in np.unique(columns, axis=0):
            idx = ma.where((columns == unique_c).all(axis=1))[0]  # ma.where is critical

            if len(idx) > 1:
                logging.debug(f"Removing columns {idx[1:]} same as {idx[0]}")
                for xi in idx[1:]:  # mask all but first occurrence
                    self.codebook[:, xi] = ma.masked
                    try:
                        self.classifiers_in_use.remove(xi)
                    except ValueError:  # if method is run twice by accident
                        pass

    def mask_low_variance_columns(self, threshold: float = 0.02):
        """Remove column *in mask only* (and from self.classifiers_in_use)."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.codebook)
        col_var_mask = selector.get_support()
        if sum(col_var_mask) > 0:
            logging.info(
                    f"Removing {sum(col_var_mask)} low variance columns w/thr {threshold}:"
                    f"{np.argwhere(col_var_mask == False).flatten()}"
            )
            for xi in np.argwhere(col_var_mask == False).flatten():
                self.codebook[:, xi] = ma.masked
                try:
                    self.classifiers_in_use.remove(xi)
                except ValueError:  # if method is run twice by accident
                    pass
            logging.debug(
                    f"Removed {np.sum(~col_var_mask)} low var columns w/thr{threshold}:"
                    f"{len(self.classifiers) - len(self.classifiers_in_use)} removed,"
                    f"{len(self.classifiers_in_use)} remaining",
            )
