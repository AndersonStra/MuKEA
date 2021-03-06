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

def tsv2feature(split):
    if split == 'train':
        infile = '../data/mscoco_imgfeat/train2014_obj36.tsv'
    elif split == 'val':
        infile = '../data/mscoco_imgfeat/val2014_obj36.tsv'
    in_data = {}
    objects_data = {}
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['image_id'] = str(int(item['image_id'].split('_')[-1]))
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            item['objects_id'] = np.frombuffer(base64.b64decode(item['objects_id'].encode()), dtype=np.int64)\
                .reshape((item['num_boxes'], -1))
            w_h = np.array([item['image_w'], item['image_h']])
            for field in ['objects_conf', 'boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field].encode()),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            spatial_feature = np.concatenate((item['boxes'][:, :2] / w_h,
                                              item['boxes'][:, 2:] / w_h), axis=1)
            in_data[item['image_id']] = {'feats': item['features'], 'sp_feats': spatial_feature}
            objects_data[item['image_id']] = {'objects': item['objects_id'].tolist(),
                                              'objects_conf': item['objects_conf'].tolist()}
    with open('../data/vqa_img_feature_%s.pickle' % split, 'wb') as f:
        pickle.dump(in_data, f)
    with open('../data/vqa_img_object_%s.json' % split,'w') as f:
        json.dump(objects_data, f, indent=4)
        
if __name__ == "__main__":
    tsv2feature('train')
    tsv2feature('val')
