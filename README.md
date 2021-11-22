# MuKEA
MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering

## Requirement
Pytorch == 1.6.0          
transformers == 3.5.0               

## Training       
1. Create model_save_dir 
```                           
mkdir model_save_dir
```

2. Prepare data

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
