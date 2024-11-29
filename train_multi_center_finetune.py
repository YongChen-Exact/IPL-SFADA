import argparse
import logging
import os
import random
import sys
import SimpleITK as sitk
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import (
    BaseDataSets1,
    RandomGenerator,
)
from networks.net_factory import net_factory
from utils import losses, ramps, utils
from val_2D import test_single_volume_fast

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/data/CY/Datasets/Prostate_B", help="Name of Experiment")  # 数据集的位置
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--model_path", type=str,
                    default="/home/data/CY/codes/BDK-SFADA/Model/MR_Prostate_A_to_Prostate_B_First_Ours_r0.2/unet_best_model.pth")
parser.add_argument("--source", type=str, default="Prostate_A", help="Source_model_name")
parser.add_argument("--target", type=str, default="Prostate_B", help="experiment_data name")
parser.add_argument("--time", type=str, default="First", help="time of finetune")
parser.add_argument("--method", type=str, default="Ours_r0.2", help="Source_model_name")
parser.add_argument("--max_iterations", type=int,
                    default=16000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=16,
                    help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1,
                    help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03,  # 0.03 for SGD   0.0001 for adam
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list,
                    default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=2023, help="random seed")
parser.add_argument("--num_classes", type=int, default=2,
                    help="output channel of network")
parser.add_argument("--load", default=False,
                    action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # The first fine-tuning calls datasets1, and the second fine-tuning is the same as normal training, using BaseDataSets
    db_train = BaseDataSets1(
        base_dir=args.root_path,
        split="train",
        transform=RandomGenerator(args.patch_size)
    )
    db_val = BaseDataSets1(base_dir=args.root_path, split="val")

    model = create_model()
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    #  Loading the original model

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.56

    iter_num = int(iter_num)
    iterator = tqdm(range(start_epoch, max_epoch), ncols=60)

    for epoch_num in iterator:
        # track mean error for entire epoch
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = (
                sampled_batch["image"],
                sampled_batch["label"],
            )
            image_batch, label_batch = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            loss = 0.5 * (ce_loss(outputs, label_batch.long(
            )) + losses.dice_loss(outputs_soft[:, 1, ...], label_batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/model_loss", loss, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume_fast(
                            sampled_batch["image"],
                            sampled_batch["label"],
                            model,
                            classes=num_classes,
                            patch_size=args.patch_size
                        )
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model_val_{}_dice".format(class_i + 1),
                        metric_list[class_i],
                        iter_num,
                    )

                performance = np.mean(metric_list)
                writer.add_scalar("info/model_val_mean_dice",
                                  performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model.pth".format(args.model))
                    utils.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_mode_path)
                    utils.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_best)

                logging.info(
                    "iteration %d : model_mean_dice: %f" % (
                        iter_num, performance)
                )

            model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                utils.save_checkpoint(
                    epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "/home/data/CY/codes/BDK-SFADA/Model/MR_{}_to_{}_{}_{}".format(
        args.source, args.target, args.time, args.method)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
