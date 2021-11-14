#!user/bin/env python
# -*- coding:utf-8 -*-
# author: DingYang time:2021/1/20
from dataset import vocab_num, my_collate
from dataset_val import KgDatasetVal
from dataset import KgDataset
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import random
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from transformers import LxmertTokenizer
from train import my_collate, cal_batch_loss, generate_tripleid, cal_acc_multi
from model import KgPreModel, tokenizer
from config import args
import json
import numpy as np


torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def my_collate_2(batch):
    id = []
    ques = []
    ans = []
    img = []
    object = []
    spatial = []
    length = []
    for item in batch:
        id.append(item['id'])
        ques.append(item['ques'])
        ans.append(item['ans'])
        img.append(item['img'])
        object.append(item['object'])
        spatial.append(item['spatial'])
        length.append(item['length'])
    res = {'id': id, 'ques': ques, 'ans': ans,
           'img': img, 'length': length,
           'object': object, 'spatial': spatial}
    return res


def write_down_prediction(object_list, qid, preds, ans, mean, length_list):
    assert len(preds) == len(qid)
    write_down_dict = {}
    mean_dict = {}
    s = 0
    for i, answer_id in enumerate(qid):
        predict_objects = []
        pred = preds[i].squeeze()
        length = length_list[i] + 36
        pred = pred[:length]
        _, idx_1 = torch.topk(pred, k=1)
        predict_objects = object_list[i][idx_1]
        # idx_3 = idx_3.tolist()
        # for idx in idx_3:
        #     try:
        #         predict_objects.append(object_list[i][idx])
        #     except IndexError:
        #         print(pred)
        #         s += 1
        write_down_dict[answer_id] = predict_objects#(predict_objects, ans[i])
        mean_dict[answer_id] = mean[i].cpu().numpy()
        # print(s)
    with open("krisp_full_generate_v1_val.json", 'w') as f:
        json.dump(write_down_dict, f, indent=4)
    with open('mean_out_temp.pickle', 'wb') as f:
        pickle.dump(mean_dict, f)


def test():
    if args.embedding:
        answer_candidate_tensor = torch.arange(0, vocab_num).view(-1, 1).long().cuda()
    # else:
    #     answer_candidate_tensor = torch.tensor(answer_embedding).float().cuda()
    #     answer_candidate_tensor = F.normalize(answer_candidate_tensor, dim=1, p=2)
    test_dataset = KgDatasetVal()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=my_collate)
    model = KgPreModel(vocab_num)
    model = model.to(device)
    model.eval()
    for epoch in range(1, 4):
        path = args.model_dir + 'model_for_epoch_%d.pth' % epoch
        print(f"\nValidation after epoch {epoch}:")
        model.load_state_dict(torch.load(path))
        preds = []
        ground_ts = []
        qids = []
        object_lists = []
        mean_outs = []
        ans_list = []
        length_list = []
        loss_state = 0
        answers = []  # [batch_answers,...]
        preds = []  # [batch_preds,...]
        preds_trip = []
        answers_trip = []
        probs = []
        ids = []
        gumbel_ids = []
        uniques_experiment= []
        embeddings = np.zeros((len(test_dataset), 300))
        idx = 0
        all_time = 0
        for batch_data in tqdm(test_dataloader):
            with torch.no_grad():
                qid = batch_data['id']
                visual_faetures = torch.tensor(batch_data['img']).float().to(device)
                source_seq = tokenizer(batch_data['ques'], padding=True, return_tensors="pt",
                                       add_special_tokens=True).to(device)
                input_id = source_seq['input_ids'].to(device)
                attention_mask = source_seq['attention_mask'].to(device)
                token_type_ids = source_seq['token_type_ids'].to(device)
                spatial_feature = torch.tensor(batch_data['spatial']).float().to(device)
                # target_true = torch.tensor(batch_data['answers_list']).to(device)
                # target_id = batch_data['answer_id']
                # most = torch.tensor(batch_data['most']).to(device)
                most_id = batch_data['mostid']

                anchor = model(input_id, attention_mask, token_type_ids, visual_faetures, spatial_feature)

                # 可视化分析代码
                # id = batch_data['id']
                # ids.extend(id)
                # target = target.squeeze()
                # p, idx_1 = torch.topk(target, dim=1, k=1)
                # gumbel_ids.extend(idx_1.squeeze().cpu().tolist())
                # embeddings[idx: idx + relation.shape[0]] = relation.cpu().numpy()
                # idx += relation.shape[0]
                anchor = F.normalize(anchor, dim=1, p=2)
                if args.embedding:
                    answer_candidate_tensor_test = model.decode_tail(answer_candidate_tensor)
                    answer_candidate_tensor_test = F.normalize(answer_candidate_tensor_test, dim=1, p=2)
                    trip_predict, _ = generate_tripleid(anchor, answer_candidate_tensor_test)
                else:
                    trip_predict, _ = generate_tripleid(anchor, answer_candidate_tensor)

                for i, pre in enumerate(most_id):
                    preds_trip.append(trip_predict[i])
                    answers_trip.append(most_id[i])
        acc_1 = 0
        acc_1_trip, ids = cal_acc_multi(answers_trip, preds_trip, return_id=True)
        print('epoch %d , acc = %f' % (
            epoch, acc_1_trip))
        with open('./online_test3500/vqa_onlinetest%d.json' % epoch, 'w') as f:
            json.dump(ids, f, indent=4)
        # np.save('best_relation_embedding.npy', embeddings)

        # 可视化分析代码
        # gumbel_dic = {}
        # for i in zip(ids, gumbel_ids):
        #     gumbel_dic[i[0]] = i[1]
        # with open('gumbel_visual_train.json','w') as f:
        #     json.dump(gumbel_dic, f, indent=4)
        # for i, id_i in enumerate(ids):
        #     gumbel_dic[id_i] = embeddings[i]
        # with open('train_relation_embedding.pickle','wb') as f:
        #     pickle.dump(gumbel_dic, f)


if __name__ == "__main__":
    test()
