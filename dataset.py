#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import json
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from config import args
from model import tokenizer
from random import sample

if args.dataset == 'krvqa':
    if args.pretrain:
        with open('data/vqa_train_filter.json','r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
        with open('data/vqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
    else:
        with open('data/kr-vqa/krvqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
        with open('data/kr-vqa/krvqa_train.json','r') as f:
            train_row = json.load(f)
    if args.accumulate:
        with open('data/krvqa-pretrain_dic_all_filter.pickle', 'rb') as f:
            a_dic = pickle.load(f)
    else:
        with open('data/kr-vqa/krvqa-ans_dic.pickle', 'rb') as f:
            a_dic = pickle.load(f)
elif args.dataset == 'okvqa':
    with open('data/vqa_img_feature_train.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    if args.pretrain:
        with open('data/vqa_train_filter.json','r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
    else:
        with open('data/okvqa_train.json','r') as f:
            train_row = json.load(f)
    if args.accumulate:
        with open('data/pretrain_dic_all_filter.pickle', 'rb') as f:
            a_dic = pickle.load(f)
    else:
        with open('data/ans_dic.pickle', 'rb') as f:
            a_dic = pickle.load(f)
elif args.dataset == 'vqav2':
    with open('data/vqa_img_feature_train.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    with open('data/vqa_train.json','r') as f:
        train_row = json.load(f)
    with open('data/vqav2/vqav2_dic_all.pickle', 'rb') as f:
        a_dic = pickle.load(f)
    with open('data/vqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature_val = pickle.load(f)
    with open('data/vqa_val.json','r') as f:
        val_row = json.load(f)
    pretrain_feature.update(pretrain_feature_val)
    train_row.update(val_row)


vocab_num = len(a_dic)
ans_all_list = a_dic.keys()
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
answers_most = []
most_answer_ids = []
neg_answer = []



n = 0


for qid, item in train_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)
    question_clean = item['question']# + answer_sentence
    questions.append(question_clean)
 


    # multi-answer
    if args.dataset == 'okvqa':
        answers.append(item['multi_answers'])
        m_ans_id = [a_dic.get(i, 0) for i in item['multi_answers']]
        most_answer_ids.append(m_ans_id)
    # most_answer.append(answer_embedding[0])


    #single answer
    else:
        answers.append(item['answer'])
        most_ans_id = a_dic.get(item['answer'], 0)
        most_answer_ids.append([most_ans_id])
    # else:
    #     most_ans_id = a_dic[most_ans]

print(len(qids))


class KgDataset(Dataset):
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

def my_collate(batch):
    batch = list(zip(*batch))
    res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
            'img': batch[3], 'spatial': batch[4],'mostid':batch[5]}
    del batch
    return res



class PretrainDataset(Dataset):
    def __init__(self, val=False, val_test=False):
        self.image_ids = image_ids
        self.qids = qids
        self.questions = questions
        self.length = question_lengths
        self.answers = answers
        self.most_answer_ids = most_answer_ids
        if val:
            self.qids = qids[30000:30500]
            self.questions = questions[30000:30500]
            self.answers = answers[30000:30500]
            self.most_answer_ids = most_answer_ids[30000:30500]
            self.image_ids = image_ids[30000:30500]
            self.length = question_lengths[30000:30500]


    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = self.qids[index]
        question = self.questions[index]
        answer = self.answers[index]

        #取特征用点小方法
        image_feature = pretrain_feature[self.image_ids[index]]['feats']
        spatial_feature = pretrain_feature[self.image_ids[index]]['sp_feats']


        # label = self.labels[index]
        # image_feature = self.img_features[index]
        # spatial_feature = self.spatial_feature[index]
        # answer_id = self.answer_ids[index]
        # answers_list = self.answers_lists[index]
        # object = self.object[index]
        most_id = self.most_answer_ids[index]
        return qid, question, answer, image_feature, spatial_feature, most_id

def my_collate_pretrain(batch):
    batch = list(zip(*batch))
    res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
            'img': batch[3], 'spatial':  batch[4],'mostid': batch[5]}
    del batch
    return res


