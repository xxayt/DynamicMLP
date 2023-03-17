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
        ''' .json 详细内容
        info: 
        images: 
            {"id": 2686843, "width": 284, "height": 222, 
            "file_name": "val/03938_Animalia_Chordata_Aves_Passeriformes_Meliphagidae_Ptilotula_penicillata/df8edd4c-fbb4-4886-8600-a429e5efac23.jpg", 
            "license": 2, "rights_holder": "megatherium", "date": "2007-10-31 00:00:00+00:00", "latitude": -21.93073, "longitude": 114.12239, 
            "location_uncertainty": null}
            {"id": 2786842, "width": 500, "height": 335, 
            "file_name": "val/03689_Animalia_Chordata_Aves_Passeriformes_Aegithalidae_Psaltriparus_minimus/01f4dc67-aac7-4cb5-b081-9c7584cf3e80.jpg", 
            "license": 1, "rights_holder": "Glenn Caspers", "date": "2018-09-18 10:52:00+00:00", "latitude": 37.8651, "longitude": -119.5388, 
            "location_uncertainty": 977}
        categories: 
            {"id": 0, "name": "Lumbricus terrestris", "common_name": "Common Earthworm", "supercategory": "Animalia", "kingdom": "Animalia", 
            "phylum": "Annelida", "class": "Clitellata", "order": "Haplotaxida", "family": "Lumbricidae", "genus": "Lumbricus", 
            "specific_epithet": "terrestris", "image_dir_name": "00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris"}
            {"id": 9999, "name": "Lygodium japonicum", "common_name": "Japanese climbing fern", "supercategory": "Plants", "kingdom": "Plantae", 
            "phylum": "Tracheophyta", "class": "Polypodiopsida", "order": "Schizaeales", "family": "Lygodiaceae", "genus": "Lygodium", 
            "specific_epithet": "japonicum", "image_dir_name": "09999_Plantae_Tracheophyta_Polypodiopsida_Schizaeales_Lygodiaceae_Lygodium_japonicum"}
        annotations: 
            {"id": 2686843, "image_id": 2686843, "category_id": 3938}
            {"id": 2786842, "image_id": 2786842, "category_id": 3689}
        licenses: 
        '''
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
            str = json.loads(str)
        for image, annotation in zip(str['images'], str['annotations']):
            img_path = os.path.join(path, image['file_name'])  # 图片路径
            label = annotation['category_id']  # 类别编号
            extra = {  # 元数据
                'date': image['date'],  # 拍摄日期时间数据
                'latitude': image['latitude'],  # 维度数据 -90 ~ 90
                'longitude': image['longitude']}   # 经度数据 -180  ~ 180
            samples.append((img_path, int(label), extra))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # 读取选定数据
        img_path, label, extra = self.samples[idx]
        date = extra['date']  # 拍摄时间
        lat = extra['latitude']  # 纬度 -90 ~ 90
        lng = extra['longitude']  # 经度 -180  ~ 180
        if (lat is None) or (lng is None) or (date is None):  # 元数据不完整
            loc = np.zeros(self.args.mlp_cin, float)
        else:
            date_time = datetime.datetime.strptime(date[:10], '%Y-%m-%d')  # 只取年月日
            # 归一化: 时间映射 -> [-1, 1]
            date = get_scaled_date_ratio(date_time)
            # 归一化: 经纬度映射 -> [-1, 1]
            lat = float(lat) / 90
            lng = float(lng) / 180
            
            # 中间编码结果 intermediate encoding result
            # Concat: channel-wise concatenation (\in R^3)
            loc = []
            if 'geo' in self.args.metadata:
                loc += [lat, lng]
            if 'temporal' in self.args.metadata:
                loc += [date]
            loc = np.array(loc)
            loc = encode_loc_time(loc)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, loc


# 正弦余弦:  (\in R^3 -> \in R^6)
def encode_loc_time(loc_time):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    feats = np.concatenate((np.sin(math.pi * loc_time), np.cos(math.pi * loc_time)))
    return feats

# 判定闰年
def _is_leap_year(year):
    if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
        return False
    return True


# 将年月日在一年中的时间映射 -> [-1, 1]
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
    # 判定每月日期是否合法
    assert day <= days[month - 1]
    sum_days = sum(days[:month - 1]) + day
    # 判定累计日期是否合法
    assert sum_days > 0 and sum_days <= total_days
    # 映射 -> [-1, 1]
    return (sum_days / total_days) * 2 - 1


# 载入训练数据
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
            transforms.RandomResizedCrop(224),  # 调整大小为224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        args=args,
    )
    # 在转为tensor前,为PIL文件,可显示图片.即删除transform=transforms后才可显示
    # print(len(dataset))  # 500000
    # print(dataset[0])
    # print(dataset[0][0])
    # dataset[0][0].save("aaa.jpg")  #展示图片

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些
    )
    # print(len(train_loader))  # 500,000 / 128 = 3906.25 -> 3907
    return train_loader

# 载入测试数据
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
