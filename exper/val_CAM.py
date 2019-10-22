import sys

sys.path.append('../')


import cv2
import cPickle
import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import my_optim
from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from utils.localization import get_topk_boxes
from utils.vistools import save_im_heatmap_box
from models import *

# default settings

ROOT_DIR = os.getcwd()
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'snapshot_bins')
IMG_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/images'))
train_list = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/list/train.txt'))
test_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test.txt'))
testbox_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test_boxes.txt'))
LR = 0.001
DISP_INTERVAL = 50
EPOCH = 100

def get_arguments():
    parser = argparse.ArgumentParser(description='DA')
    parser.add_argument("--root_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--cos_alpha", type=float, default=0.2)
    parser.add_argument("--train_root_list", type=str, default=None)
    parser.add_argument("--train_parent_list", type=str, default=None)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--test_box", type=str, default=testbox_list)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--arch", type=str, default='vgg')
    parser.add_argument("--threshold", type=str, default='0.05,0.1,0.15,0.2,0.25,0.3')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--tencrop", type=str, default='True')
    parser.add_argument("--onehot", type=str, default='False')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()


def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model, range(args.num_gpu))
    model.cuda()

    if args.resume == 'True':
        restore(args, model, None)

    return model


def val(args):

    with open(args.test_box, 'r') as f:
        gt_boxes = [map(float, x.strip().split(' ')[2:]) for x in f.readlines()]
    gt_boxes = [(box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1) for box in gt_boxes]

    # meters
    top1_clsacc = AverageMeter()
    top1_locerr = AverageMeter()
    top5_clsacc = AverageMeter()
    top5_locerr = AverageMeter()
    top1_clsacc.reset()
    top1_locerr.reset()
    top5_clsacc.reset()
    top5_locerr.reset()

    # get model
    model = get_model(args)
    model.eval()

    # get data
    _, valcls_loader, valloc_loader = data_loader(args, test_path=True)
    assert len(valcls_loader) == len(valloc_loader), \
        'Error! Different size for two dataset: loc({}), cls({})'.format(len(valloc_loader), len(valcls_loader))

    # testing
    VISLOC = True
    if VISLOC:
        # show_idxs = np.arange(20)
        np.random.seed(2333)
        show_idxs = np.arange(len(valcls_loader))
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:20]

    # evaluation classification task
    pred_prob = []
    for dat in tqdm(valcls_loader):
        # parse data
        img_path, img, label_in = dat
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label_input = label_in.repeat(10, 1)
            label = label_input.view(-1)
        else:
            label = label_in

        # forward pass
        img, label = img.cuda(), label.cuda()
        img_var, label_var = Variable(img), Variable(label)
        logits = model(img_var)

        # get classification prob
        logits0 = logits
        logits0 = F.softmax(logits0, dim=1)
        if args.tencrop == 'True':
            logits0 = logits0.view(1, ncrops, -1).mean(1)
        pred_prob.append(logits0.cpu().data.numpy())

        # update result record
        prec1_1, prec5_1 = evaluate.accuracy(logits0.cpu().data, label_in.long(), topk=(1, 5))
        top1_clsacc.update(prec1_1[0].numpy(), img.size()[0])
        top5_clsacc.update(prec5_1[0].numpy(), img.size()[0])

    pred_prob = np.concatenate(pred_prob, axis=0)
    # with open('pred_prob.pkl', 'w') as f:
    #     cPickle.dump(pred_prob, f)
    print('== cls err')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))

    # with open('pred_prob.pkl', 'r') as f:
    #     pred_prob = cPickle.load(f)
    # evaluation localization task
    thresholds = map(float, args.threshold.split(','))
    thresholds = list(thresholds)
    for th in thresholds:
        top1_locerr.reset()
        top5_locerr.reset()
        for idx, dat in tqdm(enumerate(valloc_loader)):
            # parse data
            img_path, img, label = dat

            # forward pass
            img, label = img.cuda(), label.cuda()
            img_var, label_var = Variable(img), Variable(label)
            logits = model(img_var)

            # get localization boxes
            cam_map = model.module.get_cam_maps()  # not normalized
            cam_map = cam_map.cpu().data.numpy()
            top_boxes, top_maps = get_topk_boxes(pred_prob[idx, :], cam_map, img_path[0], args.input_size,
                                                 args.crop_size, topk=(1, 5), threshold=th, mode='union')
            top1_box, top5_boxes = top_boxes

            # update result record
            locerr_1, locerr_5 = evaluate.locerr((top1_box, top5_boxes), label.cpu().data.long().numpy(), gt_boxes[idx], topk=(1, 5))
            top1_locerr.update(locerr_1, img.size()[0])
            top5_locerr.update(locerr_5, img.size()[0])
            if VISLOC:
                if idx in show_idxs:
                    save_im_heatmap_box(img_path[0], top_maps, top5_boxes, '../figs/', gt_label=label.cpu().data.long().numpy(),
                                        gt_box=gt_boxes[idx])
        print('=========== threshold: {} ==========='.format(th))
        print('== loc err')
        print('Top1: {:.2f} Top5: {:.2f}\n'.format(top1_locerr.avg, top5_locerr.avg))


if __name__ == '__main__':
    args = get_arguments()
    import json

    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
