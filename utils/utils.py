# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import datetime
import logging
import os
import pickle
import re

import numpy as np
import torch
import torch.distributed as dist
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.utils.data.sampler import Sampler


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms


def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
    """Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
    Parameters
    ----------
    X1, X2 : array or sparse matrix
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : integer
        The number of seeds to choose
    init : list
        List of points already picked
    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """

    n_samples, n_feat1 = X1.shape
    _, n_feat2 = X2.shape
    # x_squared_norms = row_norms(X, squared=True)
    centers1 = np.empty((n_clusters + len(init) - 1, n_feat1), dtype=X1.dtype)
    centers2 = np.empty((n_clusters + len(init) - 1, n_feat2), dtype=X1.dtype)

    idxs = np.empty((n_clusters + len(init) - 1,), dtype=np.long)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = init

    centers1[:len(init)] = X1[center_id]
    centers2[:len(init)] = X2[center_id]
    idxs[:len(init)] = center_id

    # Initialize list of closest distances and calculate current potential
    distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init),
                                                                                                           -1)

    candidates_pot = distance_to_candidates.sum(axis=1)
    best_candidate = np.argmin(candidates_pot)
    current_pot = candidates_pot[best_candidate]
    closest_dist_sq = distance_to_candidates[best_candidate]

    # Pick the remaining n_clusters-1 points
    for c in range(len(init), len(init) + n_clusters - 1):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(
            len(candidate_ids), -1)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        idxs[c] = best_candidate

    return None, idxs[len(init) - 1:]


def outer_product_opt(c1, d1, c2, d2):
    """Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
    """
    B1, B2 = c1.shape[0], c2.shape[0]
    t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)),
                   np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
    t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)),
                   np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
    t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
    t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
    t2 = t2.reshape(1, B2).repeat(B1, axis=0)
    return t1 + t2 - 2 * t3


# many issues with this function
def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        for key in checkpoint["state_dict"]:
            print(key)

        # size of the top layer
        N = checkpoint["state_dict"]["decoder.out_conv.bias"].size()

        # build skeleton of the model
        sob = "sobel.0.weight" in checkpoint["state_dict"].keys()
        model = models.__dict__[checkpoint["arch"]](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not "module" in key:
                return key
            return "".join(key.split(".module"))

        checkpoint["state_dict"] = {
            rename_key(key): val for key, val in checkpoint["state_dict"].items()
        }

        # load weights
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


def load_checkpoint(path, model, optimizer, from_ddp=False):
    """loads previous checkpoint

    Args:
        path (str): path to checkpoint
        model (model): model to restore checkpoint to
        optimizer (optimizer): torch optimizer to load optimizer state_dict to
        from_ddp (bool, optional): load DistributedDataParallel checkpoint to regular model. Defaults to False.

    Returns:
        model, optimizer, epoch_num, loss
    """
    # load checkpoint
    checkpoint = torch.load(path)
    # transfer state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # transfer optimizer state_dict from checkpoint to model
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # track loss
    loss = checkpoint["loss"]
    return model, optimizer, checkpoint["epoch"], loss.item()


def restore_model(logger, snapshot_path, model_num=None):
    """wrapper function to read log dir and load restore a previous checkpoint

    Args:
        logger (Logger): logger object (for info output to console)
        snapshot_path (str): path to checkpoint directory

    Returns:
        model, optimizer, start_epoch, performance
    """
    try:
        # check if there is previous progress to be restored:
        logger.info(f"Snapshot path: {snapshot_path}")
        iter_num = []
        name = "model_iter"
        if model_num:
            name = model_num
        for filename in os.listdir(snapshot_path):
            if name in filename:
                basename, extension = os.path.splitext(filename)
                iter_num.append(int(basename.split("_")[2]))
        iter_num = max(iter_num)
        for filename in os.listdir(snapshot_path):
            if name in filename and str(iter_num) in filename:
                model_checkpoint = filename
    except Exception as e:
        logger.warning(f"Error finding previous checkpoints: {e}")

    try:
        logger.info(f"Restoring model checkpoint: {model_checkpoint}")
        model, optimizer, start_epoch, performance = load_checkpoint(
            snapshot_path + "/" + model_checkpoint, model, optimizer
        )
        logger.info(f"Models restored from iteration {iter_num}")
        return model, optimizer, start_epoch, performance
    except Exception as e:
        logger.warning(f"Unable to restore model checkpoint: {e}, using new model")


def save_checkpoint(epoch, model, optimizer, loss, path):
    """Saves model as checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
    Args:
        N (int): size of returned iterator.
        images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel),
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[: self.N].astype("int")

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group["weight_decay"] * t)
        param_group["lr"] = lr


class Logger:
    """Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), "wb") as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode="inner").astype(
                np.uint8
            )
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (
                    posdis - np.min(posdis)
            ) / (np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


# set up process group for distributed computing
def distributed_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print("setting up dist process group now")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_ddp_to_nddp(state_dict):
    pattern = re.compile("module")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict
    return model_dict
