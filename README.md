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
$ mkdir data
$ cd data
```
Download annotation from 

[google drive](https://drive.google.com/file/d/1YuOUTbK7rged0gopQko5rdQuHKxiW9sv/view?usp=sharing)

We reorganized the storage structure of image features as:

```
vqa_img_feature_train.pickle{
"image_id":{'feats': features, 'sp_feats': spatial features}
}
```

The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to 1

The image features are provided by and downloaded from the original bottom-up attention' [repo](https://github.com/peteanderson80/bottom-up-attention#pretrained-features),  then follow the [script](https://github.com/AndersonStra/MuKEA/blob/main/vqa_v2_pretrain/tsv2feature.py) to process the feature.

```
python tsv2feature.py
```

### Optional download link
The image features with **objects' label** are provided by and downloaded from the origin LXMERT' [repo](https://github.com/airsplay/lxmert#google-drive), then follow the [script](https://github.com/AndersonStra/MuKEA/blob/main/vqa_v2_pretrain/tsv2feature_objects.py) to process the feature.

```
python tsv2feature_objects.py
```

### Image features for KRVQA
The image features for KRVQA are generated based on the code in this [repo](https://github.com/violetteshev/bottom-up-features), and can be downloaded form 

[google drive](https://drive.google.com/file/d/1YUhqLLXGouBsy6C-i8SIQ86VXIkclrm9/view?usp=sharing)

unzip the file and put it under `/data/kr-vqa`

### Pre-training on VQAv2
```
python train.py --embedding --model_dir model_save_dir --dataset finetune-dataset/okvqa/krvqa/vqav2 --pretrain --accumulate --validate
```       

note: `--dataset` parameter is to set the dataset for finetune


The default learning rate is set to 1e-4 which will lead to faster convergence. If the training is unstable, please set the learning rate to 1e-5 manually in the pre-training stage.

### Fine-tuning     
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa/vqav2 --load_pthpath model_save_dir/checkpoint --accumulate --validate
```

### w/o pre-training
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --validate
```

### Models

[OKVQA w/ pretrain](https://drive.google.com/file/d/1SXvdcRP6PtMM_IpAZO81xT71vU3_S16H/view?usp=share_link)

### Bibtex
```
@inproceedings{Ding2022mukea,
  title={MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering},
  author={Yang Ding and Jing Yu and Bang Liu and Yue Hu and Mingxin Cui and Qi Wug},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
