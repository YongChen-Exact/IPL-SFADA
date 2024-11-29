import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
from tqdm import tqdm
from utils.utils import get_logger
from tensorboardX import SummaryWriter
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def select(args, k, size, s_reprs, s_logits, s_inds, s_labels, t_reprs, t_logits, t_inds):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X=torch.tensor(s_reprs), y=np.array(s_labels))
    criterion = nn.KLDivLoss(reduction='none')
    kl_scores = []
    num_adv = 0
    distances = []
    for unlab_i, candidate in enumerate(
            tqdm(zip(t_reprs, t_logits), desc="Finding neighbours for every unlabeled data point")):
        distances_, neighbours = neigh.kneighbors(X=candidate[0],
                                                  return_distance=True)
        distances.append(distances_[0])
        preds_neigh = [np.argmax(s_logits[n], axis=1) for n in neighbours[0]]
        neigh_prob = [F.softmax(s_logits[n], dim=1) for n in neighbours[0]]
        pred_candidate = [np.argmax(candidate[1])]
        num_diff_pred = len(list(set(preds_neigh).intersection(pred_candidate)))
        if num_diff_pred > 0: num_adv += 1
        candidate_log_prob = F.log_softmax(candidate[1], dim=-1)
        kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
        kl_scores.append(kl.mean())

    logger.info('Total Different predictions for similar inputs: {}'.format(num_adv))
    selected_inds = np.argpartition(kl_scores, -size)[-size:]
    sampled_inds = list(np.array(t_inds)[selected_inds])
    return sampled_inds


def Savefeat(cfg):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    db_source = BaseDataSets(
        base_dir="/home/data/CY/Datasets/Prostate_A",
        split="train",
        transform=RandomGenerator([256, 256]))
    source_loader = DataLoader(db_source, batch_size=1, shuffle=False,
                               num_workers=16, pin_memory=True)
    db_target = BaseDataSets(
        base_dir="/home/data/CY/Datasets/Prostate_B",
        split="train",
        transform=RandomGenerator([256, 256]))
    target_loader = DataLoader(db_target, batch_size=1, shuffle=False,
                               num_workers=16, pin_memory=True)

    def create_model():
        # Network definition
        model = net_factory(net_type="unet", in_chns=1,
                            class_num=2)
        return model

    model = create_model()
    model.load_state_dict(
        torch.load('/home/data/CY/codes/BDK-SFADA/Model/ProstateA_unet/unet_best_model.pth')[
            "state_dict"])
    class_features = Class_Features(numbers=2)
    print("source num:", len(source_loader))
    print("target num:", len(target_loader))
    s_reprs = []
    s_logits = []
    s_inds = []
    s_labels = []

    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(source_loader):
            image_batch, label_batch, name = (
                sampled_batch["image"],
                sampled_batch["label"],
                sampled_batch["name"],
            )
            image_batch, labels = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            model.eval()
            feat_cls, output = model(image_batch)
            vectors, ids = class_features.calculate_mean_vector(feat_cls, output)
            single_image_objective_vectors = np.zeros([1, 256])
            for t in range(len(ids)):
                single_image_objective_vectors[ids[t]] = vectors[t].detach().cpu().numpy().squeeze()
            single_vectors = single_image_objective_vectors.squeeze(0).astype(float)
            s_reprs.append(single_vectors)
            s_logits.append(output.cpu())
            s_inds.append(batch_idx)
            label = label_batch.reshape(-1).detach().cpu().numpy()
            s_labels.append(label)

    t_reprs = []
    t_logits = []
    t_inds = []
    t_idx_names = {}
    t_lenth = len(target_loader)
    with torch.no_grad():
        # for epoch_num in iterator:
        for batch_idx, sampled_batch in enumerate(target_loader):
            image_batch, label_batch, name = (
                sampled_batch["image"],
                sampled_batch["label"],
                sampled_batch["name"],
            )
            image_batch, labels = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            t_idx_names[batch_idx] = name
            model.eval()
            feat_cls, output = model(image_batch)
            vectors, ids = class_features.calculate_mean_vector(feat_cls, output)
            single_image_objective_vectors = np.zeros([1, 256])
            for t in range(len(ids)):
                single_image_objective_vectors[ids[t]] = vectors[t].detach().cpu().numpy().squeeze()
            single_vectors = single_image_objective_vectors.astype(float)
            t_reprs.append(single_vectors)
            t_logits.append(output.cpu())
            t_inds.append(batch_idx)
    k = 10
    selected_rate = 0.3
    selected_num = int(t_lenth * selected_rate)
    print("selected_num:", selected_num)
    selected_list = select(args, k, selected_num, s_reprs, s_logits, s_inds, s_labels, t_reprs, t_logits, t_inds)
    selected_cases = []
    for idx in selected_list:
        selected_cases.append(str(t_idx_names[idx])[2:-2])
    file = open(os.path.join('/home/data/CY/codes/BDK-SFADA/Results/selection_list',
                             'Active_sample_Prostate_B_Ours_r0.3.txt'), 'w')
    for i in range(len(selected_cases)):
        img = str(selected_cases[i])
        img = img.strip("[]'")
        file.write(img + '\n')
    file.close()


class Class_Features:
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.all_vectors = []
        self.pred_ids = []
        self.ids = []
        self.pred_num = np.zeros(numbers + 1)
        return

    def calculate_mean_vector(self, feat_cls, outputs):
        outputs_softmax = F.softmax(outputs, dim=1)
        tensor1, tensor2 = torch.split(outputs_softmax, 1, dim=1)
        outputs_argmax = tensor1.float()
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(1):
                if scale_factor[n][t].item() == 0:
                    print('skip')
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    print('skip2')
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                s = torch.mean(s, dim=0).unsqueeze(0)
                max_pool = nn.MaxPool2d(kernel_size=16)
                output = max_pool(s)
                s = output.view(output.size(0), -1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/SPH_to_SCH_source.yml',
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    run_id = 16
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info('Let the games begin')
    Savefeat(cfg, writer, logger)
