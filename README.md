# MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering

This code implements a multimodal knowledge extraction model. The model generates output features corresponding to knowledge triplet. These knowledge features can typically be used in the following potential application scenarios:
- Model-based knowledge search. MuKEA is capable of retrieving relevant knowledge for multimodal input.
- Knowledge-based vision-language tasks, such as image caption, referring expression comprehension, vision-language navigation etc.
- Explainable deep learning, especially in the legal, medical fields.

This approach was used to achieve state-of-the-art knowledge-based visual question answering performance on [OKVQA](https://arxiv.org/abs/1906.00067) (42.59% overall accuracy) and [KRVQA](https://arxiv.org/pdf/2012.07192.pdf) (27.38% overall accuracy), as described in:

[paper link](http://arxiv.org/abs/2203.09138)

![MuKEA](https://github.com/AndersonStra/MuKEA/blob/main/model.PNG)


## Requirement
Pytorch == 1.6.0          
transformers == 3.5.0               

## Training       
1. Create model_save_dir 
```                           
mkdir model_save_dir
```

2. Preprocessing   
```
$ makedir data
$ cd data
```
follow `prepare_annotation.py` and `prepare_img.py` to process the data

### Pre-training on VQAv2
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --pretrain --accumulate --validate
```       

### Fine-tuning     
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --load_pthpath model_save_dir/checkpoint --accumulate --validate
```

### w/o pre-training
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --validate
```

### Bibtex
```
@inproceedings{Ding2022mukea,
  title={MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering},
  author={Yang Ding and Jing Yu and Bang Liu and Yue Hu and Mingxin Cui and Qi Wug},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
