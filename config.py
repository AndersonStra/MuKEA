#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--inference", action="store_true", help='complete dataset or not')
parser.add_argument("--pretrain", action="store_true", help='use vqa2.0 or not')
parser.add_argument('--batch_size', type=int, default=256,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--model_dir', type=str, default='contrasloss_check_v11/',
                    help='model file path')
parser.add_argument("--load_pthpath", default="",
                    help="To continue training, path to .pth file of saved checkpoint.")
parser.add_argument("--validate", action="store_true", help="Whether to validate on val split after every epoch.")
parser.add_argument("--embedding", action="store_true", help="Whether to train tail embedding.")
parser.add_argument("--accumulate", action="store_true", help="Whether to fine-tune.")
parser.add_argument("--dataset", default="okvqa", help="dataset that model training on")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
