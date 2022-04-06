#!/usr/bin/env python


import base64
import json
import pickle

import numpy as np
import csv
import sys
import zlib
import time
import mmap
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_h', 'image_w', 'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'num_boxes',
              'boxes', 'features']
infile = '../mscoco_imgfeat/train2014_obj36.tsv'

in_data = {}
objects_data = {}
with open(infile, "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in tqdm(reader):
        item['image_id'] = str(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        w_h = np.array([item['image_w'], item['image_h']])
        for field in ['objects_id', 'objects_conf', 'boxes', 'features']:
            item[field] = np.frombuffer(base64.b64decode(item[field].encode()),
                                        dtype=np.float32).reshape((item['num_boxes'], -1))
        spatial_feature = np.concatenate((item['boxes'][:, :2] / w_h,
                                          item['boxes'][:, 2:] / w_h), axis=1)
        in_data[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
        objects_data[item['image_id']] = {'objects': item['objects_id'].tolist(),
                                          'objects_conf': item['objects_conf'].tolist()}
with open('../data/vqa_img_feature_train.pickle', 'wb') as f:
    pickle.dump(in_data, f)
with open('../data/vqa_img_object_train.json','w') as f:
    json.dump(objects_data, f, indent=4)
