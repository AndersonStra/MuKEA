#!/usr/bin/env python


import base64
import pickle

import numpy as np
import csv
import sys
import zlib
import time
import mmap
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/train2014_resnet101_faster_rcnn_genome_36.tsv'



# Verify we can read a tsv
in_data = {}
with open(infile, "r+b") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in tqdm(reader):
        item['image_id'] = str(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        w_h = np.array([item['image_w'], item['image_h']])
        for field in ['boxes', 'features']:
            item[field] = np.frombuffer(base64.decodestring(item[field]),
                                        dtype=np.float32).reshape((item['num_boxes'], -1))
        spatial_feature = np.concatenate((item['boxes'][:, :2] / w_h,
                                          item['boxes'][:, 2:] / w_h), axis=1)
        in_data[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
np.save('data/vqa_img_feature_train.pickle', in_data)


