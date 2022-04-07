#!/usr/bin/env python


import base64
import pickle
import json
import numpy as np
import csv
import sys
import zlib
import time
import mmap
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
train_img_list = []

with open('../data/vqa_train.json', 'r') as f:
    vqa_raw = json.load(f)
for raw in vqa_raw.values():
    train_img_list.append(str(raw['image_id']))


def tsv2feature(split):
    if split == 'trainval':
        infile = '../data/trainval2014_36/trainval2014_resnet101_faster_rcnn_genome_36.tsv'
    elif split == 'test':
        infile = '../data/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
    # Verify we can read a tsv
    in_data_train = {}
    in_data_val = {}
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['image_id'] = str(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            w_h = np.array([item['image_w'], item['image_h']])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field].encode()),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            spatial_feature = np.concatenate((item['boxes'][:, :2] / w_h,
                                              item['boxes'][:, 2:] / w_h), axis=1)
            if item['image_id'] in train_img_list:
                in_data_train[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
            else:
                in_data_val[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
    print(len(in_data_train), len(in_data_val))
    assert len(in_data_train) == 82783
    assert len(in_data_val) == 40504
    if split == 'trainval':
        with open('../data/vqa_img_feature_train.pickle', 'wb') as f:
            pickle.dump(in_data_train, f)
        with open('../data/vqa_img_feature_val.pickle', 'wb') as f:
            pickle.dump(in_data_val, f)
    else:
        with open('../data/vqa_img_feature_%s.pickle' % split, 'wb') as f:
            pickle.dump(in_data_val, f)

if __name__ == "__main__":
    tsv2feature('trainval')
    # tsv2feature('val')
