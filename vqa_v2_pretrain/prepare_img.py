import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm


feature_dic = {}
feature_base = '/dataset/coco/features_36/cocobu_att'
bbox_base = '/dataset/coco/features_36/cocobu_box'
image_base = '/dataset/coco/images/train2014'
# image_base = '/datasets/test2015'
with open('/kb-vqa/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json') as f:
    a = json.load(f)
ques_list = a['questions']
for ques in tqdm(ques_list):
    image_id = str(ques['image_id'])
    if image_id not in feature_dic.keys():
        image_file = 'COCO_test2015_' + '0' * (12 - len(image_id)) + image_id +'.jpg'
        image_file = os.path.join(image_base, image_file)

        # 打开图像文件
        img = Image.open(image_file)
        img_size = img.size
        w = img_size[0]
        h = img_size[1]


        # 36 * 2048 维特征
        feature_file = image_id + '.npz'
        feature_file = os.path.join(feature_base, feature_file)
        feature_36 = np.load(feature_file)['feat']


        # bbox特征 归一化
        bbox_file = image_id + '.npy'
        bbox_path = os.path.join(bbox_base, bbox_file)
        feature_bbox = np.load(bbox_path)
        w_h = np.array([w, h])
        spatial_feature = np.concatenate((feature_bbox[:, :2] / w_h,
                                          feature_bbox[:, 2:] / w_h), axis=1)
        feature_dic[image_id] = {'feats': feature_36, 'sp_feats': spatial_feature}
with open('/kb-vqa/data/vqa_img_feature_train.pickle','wb') as f:
    pickle.dump(feature_dic, f)



