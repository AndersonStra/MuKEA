import numpy as np
from tqdm import tqdm
import pickle
new_dic={}

a = np.load('data/vqa_img_feature_train.pickle.npy', allow_pickle=True, encoding='bytes')

a = a.item()
for i in tqdm(a.keys()):
    new_dic[i.decode()] = {'feats': a[i][b'feats'], 'sp_feats': a[i][b'sp_feats']}

with open('data/vqa_img_feature_train.pickle', 'wb') as f:
    pickle.dump(new_dic, f)


