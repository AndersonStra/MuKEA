import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm

image_dic = {}
feature_dic = {}


feature_base = '/data2/yjgroup/dy/kb-vqa/data/kr-vqa/no_coco_feature'
image_base = '/data2/yjgroup/qxy/lxmert/data_qxy/Visual_Genome'
with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/image_data.json','r') as f:
    image_data = json.load(f)
for item in image_data:
    image_id = item['image_id']
    image_dic[image_id] = item

with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/krvqa_test.json','r') as f:
    test = json.load(f)

with open('/data2/yjgroup/dy/kb-vqa/data/vqa_img_feature_train.pickle', 'rb') as f:
    pretrain_feature = pickle.load(f)
with open('/data2/yjgroup/dy/kb-vqa/data/vqa_img_feature_val(all).pickle', 'rb') as f:
    pretrain_feature_val = pickle.load(f)

for qid, i in tqdm(test.items()):
    image_id = i['image_id']
    coco_id = image_dic[image_id]['coco_id']
    image_id = str(image_id)
    if coco_id is not None:
        coco_id = str(coco_id)
        if coco_id in pretrain_feature.keys():
            feature_dic[image_id] = pretrain_feature[coco_id]
        else:
            feature_dic[image_id] = pretrain_feature_val[coco_id]
    else:
        image_file = image_id + '.jpg'
        image_file = os.path.join(image_base, image_file)

        # 打开图像文件
        img = Image.open(image_file)
        img_size = img.size
        w = img_size[0]
        h = img_size[1]

        # 36 * 2048 维特征
        feature_file = image_id + '.npy'
        feature_file = os.path.join(feature_base, feature_file)
        feature = np.load(feature_file, allow_pickle=True).item()
        feature_36 = feature['features']
        feature_bbox = feature['boxes']

        w_h = np.array([w, h])
        spatial_feature = np.concatenate((feature_bbox[:, :2] / w_h,
                                          feature_bbox[:, 2:] / w_h), axis=1)
        feature_dic[image_id] = {'feats': feature_36, 'sp_feats': spatial_feature}

with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/krvqa_img_feature_test.pickle', 'wb') as f:
    pickle.dump(feature_dic, f)




