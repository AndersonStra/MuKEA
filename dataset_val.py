#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import pickle

from torch.utils.data import Dataset
import json
import numpy as np
import torch

from dataset import a_dic
from model import tokenizer
from config import args
from random import sample

if args.dataset == 'krvqa':
    with open('/kb-vqa/data/kr-vqa/krvqa_test.json','r') as f:
        val_row = json.load(f)
    with open('/kb-vqa/data/kr-vqa/krvqa_img_feature_test.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
elif args.dataset == 'okvqa':
    with open('/kb-vqa/data/okvqa_val.json','r') as f:
        val_row = json.load(f)
    with open('/kb-vqa/data/vqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
elif args.dataset == 'vqav2':
    with open('/kb-vqa/data/vqa_img_feature_test_dev.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    with open('/kb-vqa/data/vqa_test.json','r') as f:
        val_row = json.load(f)

def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sxo' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

image_ids = []
qids = []
questions = []
answers = []
labels = []
objects = []
answer_ids = []
answers_lists = []
question_lengths = []
most_answer = []
most_answer_ids = []
neg_answer = []

# with open('1_wrong.json') as f:
#     train_row = json.load(f)
n = 0


for qid, item in val_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)
    question_clean = item['question']  # + answer_sentence
    questions.append(question_clean)
    if args.dataset == 'okvqa' or args.dataset == 'vqav2':
        answers.append(item['multi_answers'])
        m_ans_id = [a_dic.get(i, -1) for i in item['multi_answers']]
        most_answer_ids.append(m_ans_id)
        # most_answer.append(answer_embedding[m_ans_id[0]])
        # item['label'].extend(question_split)
        if args.dataset == 'okvqa':
            objects.append(item['label'])
    else:
        answers.append(item['answer'])
        most_ans_id = a_dic.get(item['answer'], 0)
        most_answer_ids.append([most_ans_id])

    #if not args.inference:
#         answer_id = []
#         answer_list = torch.zeros(len(item['label'])).long()
#         for i, a in enumerate(item['label']):
#             if a == label[qid] or a == plural(label[qid]) or plural(a) == label[qid] or a[:-1] == label[qid]:
#                 answer_list[i] = 1
#                 answer_id.append(i)
#         if len(answer_id) == 0:
#             print(qid)
#             n += 1
#         answer_ids.append(answer_id)
#         answers_lists.append(answer_list)
# pickle_dic = {}
# pickle_dic['ids'] = answer_ids
# pickle_dic['ids_list'] = answers_lists
# with open('krisp_val_answer_generate_v1.pickle','wb') as f:
#     pickle.dump(pickle_dic, f)
# if not args.inference:
print(len(qids))


class KgDatasetVal(Dataset):
    def __init__(self, val=False, val_test=False):
        self.image_ids = image_ids
        self.qids = qids
        self.questions = questions
        self.answers = answers
        self.most_answer_ids = most_answer_ids

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = self.qids[index]
        question = self.questions[index]
        answer = self.answers[index]
        # 取特征用点小方法
        image_feature = pretrain_feature[self.image_ids[index]]['feats']
        spatial_feature = pretrain_feature[self.image_ids[index]]['sp_feats']
        most_id = self.most_answer_ids[index]
        return qid, question, answer, image_feature, spatial_feature, most_id


