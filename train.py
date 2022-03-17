#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bisect import bisect
from math import fabs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LxmertTokenizer

from config import args
from contrastive_loss import ContrastiveLoss, l2_sim
from dataset import KgDataset, my_collate_pretrain, PretrainDataset, my_collate
from dataset import vocab_num
from dataset_val import KgDatasetVal
from model import KgPreModel, tokenizer


# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)

# torch.manual_seed(10)
# torch.cuda.manual_seed(10)
# cudnn.benchmark = False
# cudnn.deterministic = True

torch.multiprocessing.set_sharing_strategy('file_system')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate_tripleid(batch_anchor, candidate):
    # cos distance
    similarity = batch_anchor.mm(candidate.t())   # b * v

    # l2 distance
    # similarity = l2_sim(batch_anchor, candidate)   #b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity, k=1, dim=1, largest=True)
    prob3, idx_3 = torch.topk(similarity, k=3, dim=1, largest=True)
    return idx_1.squeeze(), idx_3.squeeze()


def cal_batch_loss(target, target_true, criterion):
    target = target.view(-1, 2)
    target_true = target_true.view(-1, 1).squeeze()
    batch_loss = criterion(target, target_true)
    return batch_loss


def cal_acc_multi(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        # ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                cnt += 1
        if cnt ==1:
            acc_num += 1/3
            # ids.append([int(pred), 1])
        elif cnt == 2:
            acc_num += 2/3
            # ids.append([int(pred), 1])
        elif cnt > 2:
            acc_num += 1
            # ids.append([int(pred), 1])
        # else:
        #     ids.append([int(pred), 0])
    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num / all_num
   
def cal_acc(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                acc_num += 1
    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num / all_num


def train():
    if not args.pretrain:
        train_dataset = KgDataset(val=False)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, collate_fn=my_collate)
        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, collate_fn=my_collate)
    else:
        train_dataset = PretrainDataset(val=False)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      num_workers=8, collate_fn=my_collate_pretrain, shuffle=True)#sampler=train_sampler)
        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                         num_workers=8, collate_fn=my_collate, shuffle=False)#sampler=test_sampler)
    model = KgPreModel(vocab_num)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_graph = ContrastiveLoss(measure='dot', margin=1.0, max_violation=False)

    if args.load_pthpath == "":
        start_epoch = 0
    else:
        print('load model')
        # "path/to/checkpoint_xx.pth" -> xx
        start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

        model.module.load_state_dict(torch.load(args.load_pthpath))

    best_acc = 0
    best_epoch = 0
    best_acc_t = 0
    best_epoch_t = 0
    best_acc_t3 = 0
    if args.embedding:
        answer_candidate_tensor = torch.arange(0, vocab_num).view(-1, 1).long().cuda()
    # else:
    #     answer_candidate_tensor = torch.tensor(answer_embedding).float().cuda()
    #     answer_candidate_tensor = F.normalize(answer_candidate_tensor, dim=1, p=2)
    #model.module.load_state_dict(torch.load('contrasloss_check_v3/model_for_epoch_4.pth'))
    for epoch in range(start_epoch, args.num_epochs):
        train_answers = []
        train_preds = []
        train_preds_trip = []
        train_preds_trip_3 = []
        train_answers_trip = []
        for batch_data in tqdm(train_dataloader):
            visual_faetures = torch.tensor(batch_data['img']).float().to(device)
            source_seq = tokenizer(batch_data['ques'], padding=True, return_tensors="pt",
                                   add_special_tokens=True)
            input_id = source_seq['input_ids'].to(device)
            attention_mask = source_seq['attention_mask'].to(device)
            token_type_ids = source_seq['token_type_ids'].to(device)
            spatial_feature = torch.tensor(batch_data['spatial']).float().to(device)
            most_id = batch_data['mostid']
            most_id_tensor = torch.tensor(most_id).long().cuda()

            model.zero_grad()
            anchor = model(input_id, attention_mask, token_type_ids, visual_faetures, spatial_feature)



            if args.embedding:
                most_id_tensor = torch.tensor(most_id).view(anchor.shape[0], -1).long().cuda()
                if torch.cuda.device_count() > 1:
                    most = model.module.decode_tail(most_id_tensor)
                else:
                    most = model.decode_tail(most_id_tensor)
            else:
                most = torch.tensor(batch_data['most']).float().to(device)
            most = F.normalize(most, dim=-1, p=2)

            if args.embedding:
                if torch.cuda.device_count() > 1:
                    answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                    cls = model.module.cal_sim(anchor, answer_candidate_tensor_train)
                else:
                    answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                    cls = model.cal_sim(anchor, answer_candidate_tensor_train)
            anchor = F.normalize(anchor, dim=1, p=2)
            optimizer.zero_grad()

            most_id_tensor = most_id_tensor[:,0].squeeze()
            loss_cl = criterion_cls(cls, most_id_tensor)
            if args.dataset == 'okvqa':
                loss = 0
                for i in range(10):
                    most_i = most[:,i,:]
                    loss_mse = criterion_mse(anchor, most_i)
                    loss_graph = criterion_graph(anchor, most_i)
                    loss = loss + loss_mse + loss_graph
            else:
                loss_mse = criterion_mse(anchor, most)
                loss_graph = criterion_graph(anchor, most)
                loss = loss_mse + loss_graph

            loss_stat = loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            with torch.no_grad():
                if args.embedding:
                    if torch.cuda.device_count() > 1:
                        answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                    else:
                        answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                    answer_candidate_tensor_train = F.normalize(answer_candidate_tensor_train, dim=1, p=2)
                    trip_predict, trip_predict_3 = generate_tripleid(anchor.float(), answer_candidate_tensor_train)
                else:
                    trip_predict, trip_predict_3 = generate_tripleid(anchor.float(), answer_candidate_tensor)

                # _, idx_1 = torch.topk(cls, k=1)
                for i, pre in enumerate(most_id):
                    # train_preds.append(idx_1[i])  # [(num_nodes,)]
                    train_answers.append(most_id[i])
                    train_preds_trip.append(trip_predict[i])
                    train_preds_trip_3.append(trip_predict_3[i])
                    train_answers_trip.append(most_id[i])

        # train_acc_1 = cal_acc_old(train_answers, train_preds)
        if args.dataset == 'krvqa':
            train_acc_1_trip = cal_acc(train_answers_trip, train_preds_trip)
            print('epoch %d train_loss = %.1f, acc_trip = %.4f' % (epoch, loss_stat,
                                                                          train_acc_1_trip))
        else:
            # train_acc_1_ce = cal_acc_old(train_answers, train_preds)
            train_acc_1_trip = cal_acc_multi(train_answers_trip, train_preds_trip)
            print('epoch %d train_loss = %.1f, acc_trip = %.4f' % (epoch, loss_stat,
                                                                          train_acc_1_trip))
            # print('acc_ce = %.4f' % train_acc_1_ce)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
        else:
            torch.save(model.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
        if args.validate:
            model.eval()
            answers = []  # [batch_answers,...]
            preds = []  # [batch_preds,...]
            preds_trip = []
            preds_trip_3 = []
            answers_trip = []
            print(f"\nValidation after epoch {epoch}:")
            for i, batch_data in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    visual_faetures = torch.tensor(batch_data['img']).float().to(device)
                    source_seq = tokenizer(batch_data['ques'], padding=True, return_tensors="pt",
                                           add_special_tokens=True).to(device)
                    input_id = source_seq['input_ids'].to(device)
                    attention_mask = source_seq['attention_mask'].to(device)
                    token_type_ids = source_seq['token_type_ids'].to(device)
                    spatial_feature = torch.tensor(batch_data['spatial']).float().to(device)
                    # most = torch.tensor(batch_data['most']).to(device)
                    most_id = batch_data['mostid']


                    anchor = model(input_id, attention_mask, token_type_ids, visual_faetures, spatial_feature)
                    # if args.embedding:
                    #     answer_candidate_tensor_test = model.module.decode_tail(answer_candidate_tensor)
                    #     cls = model.module.cal_sim(anchor, answer_candidate_tensor_test)
                    anchor = F.normalize(anchor, dim=1, p=2)
                    if args.embedding:
                        if torch.cuda.device_count() > 1:
                            answer_candidate_tensor_test = model.module.decode_tail(answer_candidate_tensor)
                        else:
                            answer_candidate_tensor_test = model.decode_tail(answer_candidate_tensor)
                        answer_candidate_tensor_test = F.normalize(answer_candidate_tensor_test, dim=1, p=2)
                        trip_predict, trip_predict_3 = generate_tripleid(anchor, answer_candidate_tensor_test)
                    else:
                        trip_predict, trip_predict_3 = generate_tripleid(anchor, answer_candidate_tensor)

                    # _, idx_1 = torch.topk(cls, k=1)
                    for i, pre in enumerate(most_id):
                        # preds.append(idx_1[i])  # [(num_nodes,)]
                        answers.append(most_id[i])
                        preds_trip.append(trip_predict[i])
                        preds_trip_3.append(trip_predict_3[i])
                        answers_trip.append(most_id[i])

            # acc_1 = cal_acc_old(answers, preds)
            if args.dataset == 'krvqa':
                acc_1_trip = cal_acc(answers_trip, preds_trip)
                print('epoch %d ,  acc_trip = %.4f' % (
                    epoch, acc_1_trip))
            else:
                acc_1_trip = cal_acc_multi(answers_trip, preds_trip)
                print('epoch %d ,  acc_trip = %.4f' % (
                    epoch, acc_1_trip))
                # print('acc_ce = %.4f' % acc_1)

            if acc_1_trip > best_acc_t:
                best_acc_t = acc_1_trip
                best_epoch_t = epoch
            print("best_acc@1t={:.2%}, epoch{}".format(best_acc_t, best_epoch_t))
            if args.dataset == 'fvqa':
                print("best_acc@3t={:.2%}".format(best_acc_t3))

            model.train()



if __name__ == "__main__":
    train()
