# ESALE: Enhancing Code-Summary and Alignment Learning for Source Code Summarization

## Environment
ubuntu 18.04

### Requirements
python==3.8<br />
torch==1.7.1<br />
transformers==4.6.1<br />
tqdm==4.64.0<br />
numpy==1.22.3

## Dataset
### Original dataset
We fetch JCSD and PCSD from https://github.com/gingasan/sit3.
### Preprocessed dataset
To get the top N action words of each dataset, you can run the code as follows:
```shell
python utils/split.py \
    --dataset_name JCSD \
    --aw_cls 40
```
To get the deduplicated dataset, you can run the code as follows:
```shell
python dataset/build_JCSD_PCSD.py
python dataset/build_SiT.py
```

## ESALE

### Shared Encoder Training 
```shell
python encoder_finetune.py \
    --output_dir outputdir/ESALE \
    --dataset_name JCSD \
    --model_name_or_path  microsoft/unixcoder-base \
    --with_test \
    --with_mlm \
    --with_ulm \
    --with_awp \
    --with_cuda \
    --epochs 50
```

### Decoder Training
Since it takes too much time to generate summaries, we randomly choose 10% data from *test* dataset as *test_demo* when training a decoder.
```shell
python decoder_finetune.py \
    --output_dir outputdir/ESALE \
    --dataset_name JCSD \
    --model_name_or_path  microsoft/unixcoder-base \
    --unified_encoder_path outputdir/ESALE/unified_encoder_model/model.pth \
    --do_train \
    --do_eval \
    --do_pred \
    --with_cuda \
    --eval_steps 5000 \
    --train_steps 100000
```

### Predict
```shell
python predict.py \
    --output_dir outputdir/ESALE \
    --dataset_name JCSD \
    --model_name_or_path  microsoft/unixcoder-base \
    --unified_encoder_path outputdir/ESALE/unified_encoder_model/model.pth \
    --load_model_path outputdir/ESALE/checkpoint-best-bleu/model.bin \
    --with_cuda
```