import argparse
import os

import h5py
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/data/CY/Datasets/Prostate_D/img_unlable_Ours_r0.2", help="Name of Experiment")
parser.add_argument('--model', type=str,
                    default='unet', help='data_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--checkpoint', type=str, default="best",
                    help='last or best')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_single_volume_fast(case, net, FLAGS, test_save_path, patch_size=[256, 256], batch_size=24):
    h5f = h5py.File(FLAGS.root_path + "/{}".format(case), 'r')

    image = h5f['image'][:]
    label = h5f["label"][:]
    label[label > 0] = 1
    prediction = np.zeros_like(label)
    x, y = image.shape[0], image.shape[1]
    zoomed_slices = zoom(
        image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(zoomed_slices).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1)
        out = out.cpu().detach().numpy()
        pred = zoom(
            out, (1, x / patch_size[0], y / patch_size[1]), order=0)
        pseudo = pred.squeeze(0)
        f = h5py.File(os.path.join(test_save_path, case), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=pseudo, compression="gzip")
        f.close()


def Inference(FLAGS):
    global segmentation_performance
    image_list = sorted(os.listdir(FLAGS.root_path))

    snapshot_path = "/home/data/CY/codes/BDK-SFADA/Model/MR_Prostate_A_to_Prostate_B_First_Ours_r0.2/unet_best_model.pth"  # 微调之后的模型地址
    test_save_path = "/home/data/CY/Datasets/Prostate_D/pseudo_Ours_r0.2"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory(net_type='unet', in_chns=1,
                      class_num=FLAGS.num_classes)
    if FLAGS.checkpoint == "best":
        save_mode_path = snapshot_path
    else:
        save_mode_path = os.path.join(
            snapshot_path, 'model_iter_60000.pth')
    net.load_state_dict(torch.load(save_mode_path)["state_dict"])
    print("init weight from {}".format(save_mode_path))
    net.eval()

    for case in tqdm(image_list):
        test_single_volume_fast(case, net, FLAGS.num_classes, FLAGS, test_save_path)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(FLAGS.root_path)
    print("dice, hd95, asd    (mean-std)")
    print(metric)
