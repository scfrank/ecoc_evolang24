"""Entrypoint for running iterated learning with a codebook.

Usage:
    run_cb_itlearning.py [-C=int -R=int -D=int -M=int -E=int -G=int -T=int -N=float -o=str] [--svm_C=<str>] [--svm_L1] [--teach_by_codewords] [--teach_by_labels] [--coerce_cws] [--prune_classifiers]


Options:
    -C=<int>  number of codebook columns = classifiers  [default: 100]
    -R=<int>  number of initial codebook rows = items  [default: 5000]
    -D=<int>  representational dimensions  [default: 64]
    -M=<int>  sample M: number of items to sample to train classifiers  [default: 26]
    -E=<int>  Iterated Learning generations/epochs  [default: 10]
    -G=<int>  with synthetic: number of gold groups [default: 10]
    -T=<int>  Teacher examples (must be <= R)  [default: 100]
    -N=<int>  Noise applied to transmitted examples (float 0.0-1.0)  [default: 0.0]
    -o=<str>  output filename (.csv)  [default: out.cb_it.csv]
    --svm_C=<float>  regulariser for feature svms  [default: 1.0]
    --teach_by_codewords  teach using feature vectors/codewords  [default: False]
    --teach_by_labels  teach using cluster labels  [default: False]
    --coerce_cws  teaching examples must be valid teacher codewords  [default: False]
    --prune_classifiers  remove redundant/non-expressive features  [default: False]


"""

import csv
import logging
import pprint

import numpy as np
import numpy.random as nprand
from docopt import docopt
from scipy.spatial import distance
from sklearn import metrics

from synthetic_data import SyntheticData
from agent import CBAgent

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # fairly verbose

SEED = 26


OUTPUT_FIELDNAMES = [
    "generation",
    "items",
    "clusters",
    "features",
    # "ari",
    # "ami",
    "H",
    "C",
    "VM",
    "FS",
    "CS",
]

np.set_printoptions(precision=2)
pp = pprint.PrettyPrinter(compact=True)


def str_cl_scores(scores):
    return "  ".join([f"{k}:{v:.3f}" for k, v in scores.items()])


def str_log_line(d):
    return "  ".join(
        [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in d.items()]
    )


def evaluate_classes(labels0, pred_labels):
    """Cluster similarities between two label sets (labels are not matched)."""
    # ari = metrics.adjusted_rand_score(labels0, pred_labels)
    # ami = metrics.adjusted_mutual_info_score(labels0, pred_labels)
    homogenity = metrics.homogeneity_score(labels0, pred_labels)
    completeness = metrics.completeness_score(labels0, pred_labels)
    vmeasure = metrics.v_measure_score(labels0, pred_labels)
    scores = {
        #"ari": float(ari),
        #"ami": float(ami),
        "H": float(homogenity),
        "C": float(completeness),
        "VM": float(vmeasure),
    }

    return scores


def evaluate_features(feats0, feats1):
    """Evaluate how similar the features in feats0, feats1 are.

    Features are a set of svm classifiers (cb.classifiers[cb.classifiers_in_use])
    Similarity is max pairwise alignment.
    """
    ps0 = np.array([np.concatenate((c.coef_[0], c.intercept_)) for c in feats0])
    ps1 = np.array([np.concatenate((c.coef_[0], c.intercept_)) for c in feats1])

    distances = distance.cdist(ps0, ps1, "cosine")

    best_ds = np.min(distances, axis=1)  # this is not a 1-1 alignment necessarily!

    feature_sim = 1 - np.mean(best_ds)
    return feature_sim


def evaluate_feature_margins(feats):
    """Margin is 1/||w|| or svm.coef_[0]. """
    ms = np.array([1 / np.sqrt(np.sum(c.coef_**2)) for c in feats])
    return ms


def apply_noise_cws(cws, noise, rng):
    """Flip bits with noise probability"""
    flip_mask = np.array(
        rng.choice([0, 1], p=[1 - noise, noise], size=cws.shape), dtype=bool,
    )
    np.logical_not(cws, where=flip_mask, out=cws)
    return cws


def apply_noise_labels(labels, noise, rng):
    """Change labels (within range) with noise probability."""
    flip_mask = np.array(
        rng.choice([0, 1], p=[1 - noise, noise], size=labels.shape), dtype=bool,
    )

    # an ordered list of new labels for the flipped labels
    flipped_labels = rng.choice(np.unique(labels), np.sum(flip_mask))

    # last arg: Values to put into a where mask is True.
    np.putmask(labels, flip_mask, flipped_labels)

    if noise > 0:
        print(f"Noise changed {np.sum(flip_mask) / flip_mask.size} labels")

    return labels


def make_nextgen_codebook(
    cb0,
    ds,
    teacher_N,
    next_N,
    teach_by_codewords,
    teach_by_labels,
    noise=0.0,
    coerce_cws=False,
    prune_classifiers=False,
):
    """Args:
    ----
        cb0: current codebook/speaker
        ds: dataspace (for all users, to sample points from)
        teacher_N: number of points to get codewords from (training-labeled samples)
        next_N: number of points for next generation (all seen, not all labeled)
        teach_by_codewords: if True, label=cb0's codeword
        teach_by_codewords: if True, label=unique cluster ids
        noise: 0-1 level applied to next_cws or next_cw_labels (NB not to data points)

    Next geneneration agent/codebook is taught with a set of sampled labels
    (names or codewords) from cb0
    """
    assert teacher_N <= next_N

    # Next learner sees these datapoints (labels only used for evaluation, not teaching)
    _, next_reps, next_labels = ds.sample_points(next_N, cb0.rng)

    if teacher_N > 0:
        # NB coerce_to_codebook is necessary if we want to teach with labels.
        if teach_by_labels:
            coerce_cws = True
        next_cws, next_cw_labels = cb0.generate_codewords(
            next_reps[:teacher_N], coerce_to_codebook=coerce_cws,
        )
    else:
        next_cws = None

    teacher_labels = None
    teacher_cws = None
    if teach_by_labels:
        teacher_labels = apply_noise_labels(next_cw_labels, noise, cb0.rng)
    if teach_by_codewords:
        teacher_cws = apply_noise_cws(next_cws, noise, cb0.rng)

    num_features = len(cb0.classifiers)  # constant number of features
    if prune_classifiers:  # i.e. features = codeword length
        num_features = len(cb0.classifiers_in_use)

    next_cb = CBAgent(
        next_reps,
        next_labels,
        num_features,
        cb0.sample_M,
        cb0.classifier_accuracy_threshold,
        teacher_labels=teacher_labels,
        teacher_codewords=teacher_cws,
        svm_C=cb0.svm_C,
        rng=cb0.rng,
    )
    return next_cb


def run_iterated_learning(
    init_cb,
    ds,
    teacher_N,
    log_writer,
    num_gens=10,
    noise=0.0,
    teach_by_codewords=True,
    teach_by_labels=False,
    coerce_cws=False,
    prune_classifiers=False,
):
    curr_cb = init_cb
    last_labels = None
    last_feats = None
    for gen in range(num_gens):
        # TODO more restrictions on columns
        # low variance columns happen also when all teacher features are same
        if not teach_by_labels:
            curr_cb.mask_low_variance_columns()
        curr_cb.mask_identical_columns()  # this happens very rarely

        row_labels = curr_cb.label_identical_rows()  # "clustering"
        # For evaluation: label a consistent set of test data using the cb's rows.
        cw_preds, curr_test_labels = curr_cb.generate_codewords(
            ds.test_data, coerce_to_codebook=True,
        )

        curr_cb.update_rows(row_labels)  # why is this between evaluating row classes and test data evaluation?
        clustering_scores = curr_cb.evaluate_row_classes()

        feature_sim = 0.0
        cluster_sim = 0.0
        if last_feats is not None:
            feature_sim = evaluate_features(
                last_feats,
                [curr_cb.classifiers[i] for i in curr_cb.classifiers_in_use],
            )
            cluster_sim = evaluate_classes(last_labels, curr_test_labels)["VM"]

        last_labels = curr_test_labels
        last_feats = [curr_cb.classifiers[i] for i in curr_cb.classifiers_in_use]

        d = {
            "generation": gen,
            "items": len(row_labels),
            "clusters": len(np.unique(row_labels)),
            "features": len(curr_cb.classifiers_in_use),
            **clustering_scores,
            "FS": float(feature_sim),
            "CS": float(cluster_sim),  # VM btw prev & current cluster
        }

        log_writer.writerow(d)
        print(str_log_line(d))

        next_cb = make_nextgen_codebook(
            curr_cb,
            ds,
            teacher_N=teacher_N,
            next_N=len(row_labels),
            teach_by_codewords=teach_by_codewords,
            teach_by_labels=teach_by_labels,
            noise=noise,
            coerce_cws=coerce_cws,
            prune_classifiers=prune_classifiers,
        )
        curr_cb = next_cb


def main(clopt):

    logging.info(f"CL Config {clopt}")

    log_file = clopt["-o"]
    log_writer = csv.DictWriter(
        open(log_file, "w", newline=""), fieldnames=OUTPUT_FIELDNAMES,
    )
    log_writer.writeheader()

    rng = nprand.default_rng(SEED)

    rep_dim = int(clopt["-D"])  # number of dimensions of data representations

    ds = SyntheticData(
        rep_dim,  # number of dimensions
        int(clopt["-G"]),  # number of gold labels
        cluster_std=1,  # 1 = separable  # 4, in/less seperable
        rng=rng.integers(low=0, high=200),
    )

    ds.setup_representations(rep_dim=rep_dim, rep_type=None)

    # sample num_rows number of points from the full dataset
    num_rows = int(clopt["-R"])
    if num_rows > 0:
        data, reps, labels = ds.sample_points(num_rows, rng)
    else:
        data, reps, labels = ds.representations, ds.labels

    # first generation, random columns
    cb0 = CBAgent(
        reps,
        labels,
        num_classifiers=int(clopt["-C"]),
        sample_M=int(clopt["-M"]),
        teacher_labels=None,
        teacher_codewords=None,
        svm_C=float(clopt["--svm_C"]),
        rng=rng,
    )

    run_iterated_learning(
        cb0,
        ds,
        int(clopt["-T"]),
        log_writer,
        int(clopt["-E"]),
        noise=float(clopt["-N"]),
        teach_by_codewords=bool(clopt["--teach_by_codewords"]),
        teach_by_labels=bool(clopt["--teach_by_labels"]),
        coerce_cws=bool(clopt["--coerce_cws"]),
        prune_classifiers=bool(clopt["--prune_classifiers"]),
    )
    return log_file


if __name__ == "__main__":
    clopt = docopt(__doc__)  # parse arguments based on docstring above
    main(clopt)
