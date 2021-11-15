#!user/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel

from attention import MultiHeadAttention, attention
from prepare.gumbel_softmax import gumbel_softmax

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
model = LxmertModel.from_pretrained('./premodel', config=config)
# model.resize_token_embeddings(len(tokenizer))
config.return_dict=True




class KgPreModel(nn.Module):
    def __init__(self, vocab_num):
        super(KgPreModel, self).__init__()
        self.vocab_num = vocab_num
        self.PreLayer = model
        # self.linear = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 1))  # 有的模型有这个参数
        self.linear_vision = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        self.linear_300 = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        self.linear_classifytask = nn.Linear(300, 1024)   # 有的模型有这个参数
        self.tail_decode = nn.Embedding(vocab_num, 300)
        init.uniform_(self.tail_decode.weight.data)
        # self.sa = MultiHeadAttention(8, 768)
        self.v_att_proj = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)

        # self.v_att_proj = nn.Linear(2048, 2048)
        # self.cls_att_proj = nn.Linear(768, 2048)
        # self.v_cls_att_value = nn.Linear(2048, 1)
        # self.linear_classifytask = nn.Sequential(nn.Linear(768 + 768, 2048), nn.ReLU(), nn.Linear(2048, vocab_num))

        # self.h_trans = nn.Linear(300, 300)
        # self.r_trans = nn.Linear(300, 300)
        # self.t_trans = nn.Linear(300, 300)




    def forward(self, i, a, t, v, s):
        bert_output = self.PreLayer(i, attention_mask=a,
                                    token_type_ids=t, visual_feats=v, visual_pos=s)
        language_output = bert_output.language_output[:,1:-1]
        vision_output = bert_output.vision_output
        cls = bert_output.pooled_output

        sum_vision = self.linear_vision(cls)

        # 原始gumbel-softmax(abandon)
        # lv_output = torch.cat((vision_output, language_output), dim=1)    # b * length * 768
        # kg_output = self.linear(lv_output)    # b * length * 1

        # affinity matrix
        l_att = self.l_att_proj(language_output)
        v_att = self.v_att_proj(vision_output)
        sim_matrix_v2l = torch.matmul(v_att, l_att.transpose(1,2))  # b * v_length * l_length
        kg_output, k = torch.topk(sim_matrix_v2l, dim=-1, k=1)

        # #normalize(abandon)
        # kg_output = F.log_softmax(kg_output,dim=-1)



        # # self attention
        # sum_vision = self.sa(lv_output, lv_output, lv_output)
        # length = lv_output.shape[1]
        # # cls = cls.unsqueeze(1)
        # # sum_vision, alpha = attention(cls, lv_output, lv_output)
        # sum_vision = sum_vision.sum(dim=1) / length
        # sum_vision = self.linear_vision(sum_vision)


        # hard attention
        hard_attention_value = gumbel_softmax(kg_output.squeeze())
        head = (vision_output * hard_attention_value.unsqueeze(-1)).sum(-2)

        # soft attention
        # kg_output = F.softmax(kg_output.squeeze(), dim=-1)
        # head = (vision_output * kg_output.unsqueeze(-1)).sum(-2)

        head_300 = self.linear_300(head)

        # other embedding method(abandon)
        # self.r_trans_ = self.r_trans(sum_vision)  # b*300
        # h_trans = self.h_trans(head_300)          # b*300
        # project_h = torch.matmul(self.r_trans_.unsqueeze(-1), h_trans.unsqueeze(-2)) + torch.eye(300).cuda()  # b*300*300
        # head_300 = torch.matmul(head_300.unsqueeze(1), project_h).squeeze()
        anchor = sum_vision + head_300


        return anchor

    def decode_tail(self, most):
        # most = self.linear_tail(most)
        # # most = self.tanh(most)
        # most = self.linear_tail2(most)
        most = self.tail_decode(most).squeeze()

        # t_trans = self.t_trans(most).squeeze()
        # if t_trans.shape[0] != self.r_trans_.shape[0]:
        #     project_t = torch.matmul(t_trans.unsqueeze(-2).repeat(1, self.r_trans_.shape[0], 1).transpose(1,2), self.r_trans_)\
        #                 + torch.eye(300).cuda()
        # else:
        #     project_t = torch.matmul(self.r_trans_.unsqueeze(-1), t_trans.unsqueeze(-2)) + torch.eye(300).cuda()
        # most = torch.matmul(most.unsqueeze(1), project_t).squeeze()

        return most.squeeze()

    def cal_sim(self, anchor, most):
        anchor = self.linear_classifytask(anchor)  # b * 1024
        most = self.linear_classifytask(most)     # vocab_num * 1024
        sim_out = anchor.mm(most.t())

        return sim_out.squeeze()
