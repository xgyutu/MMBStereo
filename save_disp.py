import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import skimage
import skimage.io
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='MMB-Stereo: Mamba-based Multi-branch Cost Aggregation - Disparity Saving')
parser.add_argument('--model', default='MMBStereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='/root/autodl-tmp/KITTI/KITTI_2015/', help='KITTI 2015 data path')
parser.add_argument('--kitti12_datapath', default='/root/autodl-tmp/KITTI/KITTI_2012/', help='KITTI 2012 data path')
parser.add_argument('--testlist', default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./runs/kitti/MMB-Stereo/checkpoint_000499.ckpt', help='load weights from specific checkpoint')
args = parser.parse_args()

StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

save_dir = './output/kitti15/disp_0'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
