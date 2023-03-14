import datetime
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class INatDataset(Dataset):
    def __init__(self, dataname, path, train, transform=None, args=None):
        self.transform = transform
        self.args = args

        if train:
            if 'mini' in dataname:
                jpath = os.path.join(path, 'train_mini.json')
            else:
                jpath = os.path.join(path, 'train.json')
        else:
            jpath = os.path.join(path, 'val.json')

        samples = []
        with open(jpath, 'r') as f:
            str = f.read()
            annotations = json.loads(str)
        for img, ann in zip(annotations['images'], annotations['annotations']):
            img_path = os.path.join(path, img['file_name'])
            label = ann['category_id']
            extra = {'date': img['date'], 'latitude': img['latitude'], 'longitude': img['longitude']}
            samples.append((img_path, int(label), extra))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # 每次怎么读数据
        img_path, label, extra = self.samples[idx]
        date = extra['date']  # 拍摄时间
        lat = extra['latitude']  # 纬度 -90 ~ 90
        lng = extra['longitude']  # 经度 -180  ~ 180
        if (lat is not None) and (lng is not None) and (date is not None):
            date_time = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
            date = get_scaled_date_ratio(date_time)
            lat = float(lat) / 90
            lng = float(lng) / 180
            loc = []
            if 'geo' in self.args.metadata:
                loc += [lat, lng]
            if 'temporal' in self.args.metadata:
                loc += [date]
            loc = np.array(loc)
            loc = encode_loc_time(loc)
        else:
            loc = np.zeros(self.args.mlp_cin, float)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, loc


def encode_loc_time(loc_time):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    feats = np.concatenate((np.sin(math.pi * loc_time), np.cos(math.pi * loc_time)))
    return feats


def _is_leap_year(year):
    if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
        return False
    return True


def get_scaled_date_ratio(date_time):
    r'''
    scale date to [-1,1]
    '''
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total_days = 365
    year = date_time.year
    month = date_time.month
    day = date_time.day
    if _is_leap_year(year):
        days[1] += 1
        total_days += 1

    assert day <= days[month - 1]
    sum_days = sum(days[:month - 1]) + day
    assert sum_days > 0 and sum_days <= total_days

    return (sum_days / total_days) * 2 - 1


def load_train_dataset(args):
    if args.data == 'inat17':
        args.num_classes = 5089
    elif args.data == 'inat18':
        args.num_classes = 8142
    elif args.data == 'inat21_mini' or 'inat21_full':
        args.num_classes = 10000
    else:
        raise NotImplementedError

    dataset = INatDataset(
        dataname=args.data,
        path=args.data_dir,
        train=True,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        args=args,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
        pin_memory=True,  # 生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些
    )
    return train_loader


def load_val_dataset(args):
    if args.data == 'inat17':
        args.num_classes = 5089
    elif args.data == 'inat18':
        args.num_classes = 8142
    elif args.data == 'inat21_mini' or 'inat21_full':
        args.num_classes = 10000
    else:
        raise NotImplementedError
    '''
    fivecrop 就是在原图片的四个角和中心各截取一幅大小为 size 的图片
    而 tencrop 就是在 fivecrop 基础上再进行水平或者竖直翻转(flip)，默认为水平翻转。
    '''
    if args.tencrop:
        dataset = INatDataset(
            dataname=args.data,
            path=args.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
            ]),
            args=args,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        dataset = INatDataset(
            dataname=args.data,
            path=args.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            args=args,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=args.num_workers,
            pin_memory=True,
        )
    return val_loader
