import sys

sys.path.append('../')


import torch
from torch import optim
import argparse
import os
import time
import shutil
import json
import datetime
from torch.autograd import Variable
from visdom import Visdom

import numpy as np
import my_optim
from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from tensorboardX import SummaryWriter
from models import *

# default settings
ROOT_DIR = os.getcwd()
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'snapshot_bins')
IMG_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/images'))
train_list = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/list/train.txt'))
train_root_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/order_label.txt'))
train_parent_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/family_label.txt'))

test_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test.txt'))

LR = 0.001
EPOCH = 21
DISP_INTERVAL = 20


def get_arguments():
    parser = argparse.ArgumentParser(description='SPG')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                        help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default=IMG_DIR,
                        help='Directory of training images')
    parser.add_argument("--cos_alpha", type=float, default=0.2)
    parser.add_argument("--num_maps", type=float, default=8)
    parser.add_argument("--vis_name", type=str, default='')
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--train_root_list", type=str, default=train_root_list)
    parser.add_argument("--train_parent_list", type=str, default=train_parent_list)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--arch", type=str, default='vgg_DA')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--diff_lr", type=str, default='True')
    parser.add_argument("--decay_points", type=str, default='80')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--log_dir", type=str, default='../log')
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='False')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def get_model(args):
    model = eval(args.arch).model(pretrained=True,
                                  num_classes=args.num_classes,
                                  args=args)
    model.cuda()
    model = torch.nn.DataParallel(model, range(args.num_gpu))

    lr = args.lr
    added_layers = ['fc', 'cls'] if args.diff_lr == 'True' else []
    weight_list = []
    bias_list = []
    added_weight_list = []
    added_bias_list = []
    print('\n following parameters will be assigned 10x learning rate:')
    for name, value in model.named_parameters():
        if any([x in name for x in added_layers]):
            print name
            if 'weight' in name:
                added_weight_list.append(value)
            elif 'bias' in name:
                added_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    optimizer = optim.SGD([{'params': weight_list, 'lr': lr},
                           {'params': bias_list, 'lr': lr * 2},
                           {'params': added_weight_list, 'lr': lr * 10},
                           {'params': added_bias_list, 'lr': lr * 20}],
                          momentum=0.9, weight_decay=0.0005, nesterov=True)

    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return model, optimizer


def train(args):
    batch_time = AverageMeter()
    lossCos = AverageMeter()
    losses = AverageMeter()
    loss_root = AverageMeter()
    loss_parent = AverageMeter()
    loss_child = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_parent = AverageMeter()
    top5_parent = AverageMeter()
    top1_root = AverageMeter()
    top5_root = AverageMeter()
    model, optimizer = get_model(args)
    model.train()
    train_loader, _, _ = data_loader(args)

    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch \t loss \t pred@1 \t pred@5\n')

    # construct writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)
    while current_epoch < total_epoch:
        model.train()
        lossCos.reset()
        losses.reset()
        loss_root.reset()
        loss_parent.reset()
        loss_child.reset()
        top1.reset()
        top5.reset()
        top1_parent.reset()
        top5_parent.reset()
        top1_root.reset()
        top5_root.reset()

        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)

        if res:
            with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
                for g in optimizer.param_groups:
                    out_str = 'Epoch:%d, %f\n' % (current_epoch, g['lr'])
                    fw.write(out_str)

        steps_per_epoch = len(train_loader)
        for idx, dat in enumerate(train_loader):
            img_path, img, label = dat
            global_counter += 1
            img, root_label, parent_label, child_label = img.cuda(), label[0].cuda(), label[1].cuda(), label[2].cuda()
            img_var, root_label_var, parent_label_var, child_label_var = Variable(img), Variable(root_label), Variable(
                parent_label), Variable(child_label)

            logits = model(img_var)
            loss_val, loss_root_val, loss_parent_val, loss_child_val, lossCos_val = model.module.get_loss(logits, root_label_var, parent_label_var, child_label_var)

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)

            # network parameter update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if not args.onehot == 'True':
                logits5 = torch.squeeze(logits[-1])
                prec1, prec5 = evaluate.accuracy(logits5.data, child_label.long(), topk=(1, 5))
                top1.update(prec1[0], img.size()[0])
                top5.update(prec5[0], img.size()[0])
                logits4 = torch.squeeze(logits[-2])
                prec1_4, prec5_4 = evaluate.accuracy(logits4.data, parent_label.long(), topk=(1, 5))
                top1_parent.update(prec1_4[0], img.size()[0])
                top5_parent.update(prec5_4[0], img.size()[0])
                logits3 = torch.squeeze(logits[-3])
                prec1_3, prec5_3 = evaluate.accuracy(logits3.data, root_label.long(), topk=(1, 5))
                top1_root.update(prec1_3[0], img.size()[0])
                top5_root.update(prec5_3[0], img.size()[0])

            losses.update(loss_val.data, img.size()[0])
            loss_root.update(loss_root_val.data, img.size()[0])
            loss_parent.update(loss_parent_val.data, img.size()[0])
            loss_child.update(loss_child_val.data, img.size()[0])
            lossCos.update(lossCos_val.data, img.size()[0])
            batch_time.update(time.time() - end)

            end = time.time()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA
                eta_seconds = ((total_epoch - current_epoch) * steps_per_epoch +
                               (steps_per_epoch - idx)) * batch_time.avg
                eta_str = "{:0>8}".format(datetime.timedelta(seconds=int(eta_seconds)))
                eta_seconds_epoch = steps_per_epoch * batch_time.avg
                eta_str_epoch = "{:0>8}".format(datetime.timedelta(seconds=int(eta_seconds_epoch)))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ETA {eta_str}({eta_str_epoch})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss cos {lossCos.val:.4f} ({lossCos.avg:.4f})\t'
                      'Loss root {loss_parent.val:.4f} ({loss_root.avg:.4f})\t'
                      'Loss parent {loss_parent.val:.4f} ({loss_parent.avg:.4f})\t'
                      'Loss child {loss_child.val:.4f} ({loss_child.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'parent: Prec@1 {top1_parent.val:.3f} ({top1_parent.avg:.3f})\t'
                      'Prec@5 {top5_parent.val:.3f} ({top5_parent.avg:.3f})\t'
                      'root: Prec@1 {top1_root.val:.3f} ({top1_root.avg:.3f})\t'
                      'Prec@5 {top5_root.val:.3f} ({top5_root.avg:.3f})'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader), batch_time=batch_time,
                    eta_str=eta_str, eta_str_epoch=eta_str_epoch, loss=losses, loss_root=loss_root,
                    loss_parent=loss_parent, loss_child=loss_child,
                    top1=top1, top5=top5, top1_parent=top1_parent, top5_parent=top5_parent, top1_root=top1_root,
                    top5_root=top5_root, lossCos = lossCos,))

        plotter.plot('rootLoss', 'train', current_epoch, loss_root.avg)
        plotter.plot('childLoss', 'train', current_epoch, loss_child.avg)
        plotter.plot('parentLoss', 'train', current_epoch, loss_parent.avg)
        plotter.plot('cosLoss', 'train', current_epoch, lossCos.avg)
        plotter.plot('top1', 'train', current_epoch, top1.avg)
        plotter.plot('top5', 'train', current_epoch, top5.avg)
        plotter.plot('parent Top1', 'train', current_epoch, top1_parent.avg)
        plotter.plot('parent Top5', 'train', current_epoch, top5_parent.avg)
        plotter.plot('root Top1', 'train', current_epoch, top1_root.avg)
        plotter.plot('root Top5', 'train', current_epoch, top5_root.avg)

        current_epoch += 1
        if current_epoch % 50 == 0:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'arch': 'resnet',
                                'global_counter': global_counter,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d_glo_step_%d.pth.tar'
                                     % (args.dataset, current_epoch, global_counter))

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            fw.write('%d \t %.4f \t  %.4f \t  %.4f \t %.4f \t  %.4f \t %.3f \t %.3f\t %.3f \t %.3f\t %.3f \t %.3f\n' % (current_epoch, losses.avg, lossCos.avg, loss_root.avg, loss_parent.avg, loss_child.avg, top1_root.avg, top5_root.avg, top1_parent.avg, top5_parent.avg, top1.avg, top5.avg))



        losses.reset()
        loss_root.reset()
        loss_parent.reset()
        loss_child.reset()
        top1.reset()
        top5.reset()
        top1_parent.reset()
        top5_parent.reset()
        top1_root.reset()
        top5_root.reset()
        lossCos.reset()

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name, update='append')
    def plot_heatmap(self, map, epoch):
        self.viz.heatmap(X = map,
                         env=self.env,
                         opts=dict(
                                    title='activations {}'.format(epoch),
                                    xlabel='modules',
                                    ylabel='classes'
                                ))

if __name__ == '__main__':
    args = get_arguments()
    global plotter
    plotter = VisdomLinePlotter(env_name=args.vis_name)
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
