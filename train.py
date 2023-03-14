import argparse
import datetime
import os
import random
import time
from timm.utils import accuracy, AverageMeter
import numpy as np
import torch
import torch.optim as optim

import dataset
import models
from utils import create_logging, LabelSmoothingLoss, save_checkpoint, adjust_learning_rate, accuracy, mixup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--save_dir', default='./outputs', type=str)
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--random_seed', default=37, type=int)
    # data
    parser.add_argument('--data', default='inat21_mini', type=str, help='inat21_mini|inat21_full')
    parser.add_argument('--data_dir', default='datasets/iNat2021', type=str)
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--metadata', default='geo_temporal', type=str, help='geo_temporal|geo|temporal')
    # train
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--start_lr', default=0.04, type=float)
    parser.add_argument('--stop_epoch', default=90, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # model
    parser.add_argument('--model_file', default='sk2res2net_dynamic_mlp', type=str, help='model file name')
    parser.add_argument('--model_name', default='sk2res2net101', type=str, help='model type in detail')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', default='latest', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    # dynamic MLP
    parser.add_argument('--mlp_type', default='c', type=str, help='dynamic mlp versions: a|b|c')
    parser.add_argument('--mlp_d', default=256, type=int)
    parser.add_argument('--mlp_h', default=64, type=int)
    parser.add_argument('--mlp_n', default=2, type=int)

    args = parser.parse_args()
    args.mlp_cin = 0
    if 'geo' in args.metadata:
        args.mlp_cin += 4
    if 'temporal' in args.metadata:
        args.mlp_cin += 2
    return args

def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s_train.log' % creat_time))  # 创建训练保存log文件

    # get datasets
    train_loader = dataset.load_train_dataset(args)  # 加载训练数据集
    val_loader = dataset.load_val_dataset(args)  # 加载测试数据集

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get net
    logger.info(f"Creating model:{args.model_file} -> {args.model_name}")
    model = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)  # 从mode_file中找到对应model_name的模型
    model.cuda()
    model = torch.nn.DataParallel(model)
    # logger.info(model)  # 打印网络结构

    # get criterion 损失函数
    criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).cuda()
    # get optimizer 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 1
    max_accuracy = 0.0
    # 如果之前有与训练权重，直接作为基础恢复训练
    if args.resume:
        if args.resume in ['best', 'latest']:
            args.resume = os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            state_dict = torch.load(args.resume)
            if 'model' in state_dict:
                start_epoch = state_dict['epoch'] + 1
                model.load_state_dict(state_dict['model'])
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
            else:
                model.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
            if 'max_accuracy' in state_dict:
                max_accuracy = state_dict['max_accuracy']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        epoch = start_epoch - 1
        acc1, acc5, outputs = validate(val_loader, model, criterion, epoch, logger, args)
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)
        return

    best_acc1 = 0.0
    best_acc5 = 0.0
    args.time_sec_tot = 0.0
    args.start_epoch = start_epoch
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.stop_epoch + 1):
        # 训练
        train_one_epoch_local_data(train_loader, model, criterion, optimizer, epoch, logger, args)
        save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='latest')
        # 测试
        logger.info(f"**********latest test***********")
        acc1, acc5, loss = validate(val_loader, model, criterion, epoch, logger, args)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')
        if acc1 > best_acc1:
            best_acc1, best_acc5 = acc1, acc5
            save_checkpoint(epoch, model, optimizer, max_accuracy, args, logger, save_name='best')
        logger.info('Exp path: %s' % args.path_log)
    # 总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_local_data(train_loader, model, criterion, optimizer, epoch, logger, args):
    model.train()
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()  # 自动混合精度训练

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for iter, (images, target, location) in enumerate(train_loader):
        # change learning rate
        learning_rate = adjust_learning_rate(optimizer, iter, epoch, num_steps, args)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        images, target_a, target_b, lam, index = mixup(images, target, alpha=0.4)
        location = lam * location + (1 - lam) * location[index]

        # compute output
        with torch.cuda.amp.autocast():
            if args.image_only:
                output = model(images)
            else:
                output = model(images, location)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        # measure accuracy and record loss
        acc1, acc5 = lam * accuracy(output, target_a, topk=(1, 5)) + (1 - lam) * accuracy(output, target_b, topk=(1, 5))
        # compute gradient and do sgd step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 储存batch_time和loss
        batch_time.update(time.time() - end)  # 记录每次迭代batch所需时间
        end = time.time()
        loss_meter.update(loss.item(), target.size(0))  # target.size(0)
        # log输出训练参数
        if iter % 300 == 0:
            etas = batch_time.avg * (num_steps - iter)
            # lr = optimizer.param_groups[0]['lr']
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{args.stop_epoch}][{iter}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {learning_rate:.8f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc@1: {acc1.item():.4f}\t'
                f'acc@5: {acc5.item():.4f}\t')
            # logger.info('\t'.join(outputs))
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate(val_loader, model, criterion, epoch, logger, args):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    for iter, (images, target, location) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()
        # compute output
        with torch.no_grad():
            if args.image_only:
                output = model(images)
            else:
                output = model(images, location)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # 更新记录
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # log输出测试参数
    if iter % 200 == 0:
        logger.info(
            f'Test: [{iter}/{len(val_loader)}]\t'
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
            f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args = parse_option()

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
