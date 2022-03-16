# MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering

This code implements a multimodal knowledge extraction model. The model generates output features corresponding to knowledge triplet. These knowledge features can typically be used in the following potential application scenarios:
- Model-based knowledge search. MuKEA is capable of retrieving relevant knowledge for multimodal input.
- Knowledge-based vision-language tasks, such as image caption, referring expression comprehension, vision-language navigation etc.
- Explainable deep learning, especially in the legal, medical fields.

This approach was used to achieve state-of-the-art knowledge-based visual question answering performance on OKVQA (42.59% overall accuracy) and KRVQA (27.38% overall accuracy), as described in:

(paper link)

![MuKEA](model.png "model")

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
$ makedir kb-vqa
$ cd kb-vqa
$ makedir data
$ cd data
```
follow `prepare_annotation.py` and `prepare_img.py` to process the data

### Pre-training on VQAv2
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --pretrain --accumulate --validate
```
The parameter of dataset only determines the dataset to test and does not affect the result of pre-training         

### Fine-tuning     
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --load_pthpath model_save_dir/checkpoint --accumulate --validate
```

### w/o pre-training
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa --validate
```
