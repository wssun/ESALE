# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import logging
import os
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, RobertaConfig, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup)

import utils.bleu as bleu
from utils.bert_dataset import BertDataset, get_instances
from models.Seq2Seqs import Seq2Seq, Seq2Seq4unixcoder

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.") # required
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="The name of dataset. It should be JCSD or PCSD.")
    parser.add_argument("--model_name_or_path", default= None, type=str, required=True,
                        help="Path to pre-trained model: e.g. microsoft/codebert-base or microsoft/unixcoder-base")
    parser.add_argument("--unified_encoder_path", default=None, type=str, 
                        help="Path to unified_encoder: Should contain the .pth files" )

    # Other parameters
    parser.add_argument("--max_seq", type=int,
                        default=256, help="maximum input sequence len")
    parser.add_argument("--max_output_seq", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pred", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--with_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--cuda_devices", type=str, nargs="+",
                        default="0", help="CUDA device ids")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=5, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--eval_steps", default=5000, type=int,
                        help="")
    parser.add_argument("--train_steps", default=100000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or not args.with_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.with_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    if args.model_name_or_path == "microsoft/unixcoder-base":
        config.is_decoder = True
    padding_id = tokenizer.convert_tokens_to_ids("<pad>")
    # budild model
    if args.unified_encoder_path:
        encoder = torch.load(args.unified_encoder_path) # required
        encoder = encoder.encoder
    else:
        encoder = RobertaModel.from_pretrained(args.model_name_or_path,config = config)

    if args.model_name_or_path == "microsoft/codebert-base":
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size, max_length=args.max_output_seq, 
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id, padding_id = padding_id)
    
    else:
        # in unixcoder, decoder == encoder
        model = Seq2Seq4unixcoder(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_output_seq,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    model.to(device)
    if args.n_gpu>1:
        model = nn.DataParallel(model,device_ids=args.cuda_devices)

    # Prepare data loader
    my_dir = os.path.join("data",args.dataset_name)
    instances = []
    # for file_type in ["train","valid","test_demo"]:
    for file_type in ["demo","demo","demo"]: 
        code_path = os.path.join(my_dir, file_type, "code.txt")
        NL_path = os.path.join(my_dir, file_type, "nl.txt")
        if args.model_name_or_path == "microsoft/codebert-base":
            instance = get_instances(
                code_path, NL_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask=True, tokenizer=tokenizer, SOS = "<s>")
        else:
            instance = get_instances(
                code_path, NL_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask=True, tokenizer=tokenizer, SOS = "<mask0>")
        instances.append(instance)
    train_dataset = BertDataset(instances[0])
    test_dataset = BertDataset(instances[1])
    pred_demo_dataset = BertDataset(instances[2])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    pred_demo_dataloader = DataLoader(pred_demo_dataset, batch_size=args.batch_size)
    if args.do_train:
        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps *
                    args.batch_size//len(train_dataset))

        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 0 # require
        bar = tqdm(range(num_train_optimization_steps),
                   total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            data = {key: value.to(device) for key, value in batch.items()}
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

            loss, _, _, _ = model(
                    source_ids = data["code_ids"], target_ids = data["comments_ids"])
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(
                tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += data["code_ids"].size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0 
                eval_flag = False
                
                # Start Evaling model
                model.eval()
                logger.info("***** Running evaluating *****")
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    data = {key: value.to(device) for key, value in batch.items()}
                    with torch.no_grad():
                        _, loss, num, _ = model(
                            source_ids = data["code_ids"], target_ids = data["comments_ids"])
                        eval_loss += loss.sum().item()
                        tokens_num += num.sum().item()
                    
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {"eval_ppl": round(np.exp(eval_loss), 5),
                          "global_step": global_step+1,
                          "train_loss": round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)

                # save last checkpoint
                last_output_dir = os.path.join(
                    args.output_dir, "checkpoint-last")
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(
                    model, "module") else model  # Only save the model it-self
                output_model_file = os.path.join(
                    last_output_dir, "model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  "+"*"*20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-best-ppl")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(
                        model, "module") else model  # Only save the model it-self
                    output_model_file = os.path.join(
                        output_dir, "model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                else:
                    logger.info("  This epoch is not Best ppl:%s", round(np.exp(best_loss), 5))
                    logger.info("  "+"*"*20)

                if args.do_pred:
                    # Calculate bleu
                    model.eval()
                    logger.info("***** Running calculating bleu *****")
                    candidates = []
                    references = []
                    for batch in tqdm(pred_demo_dataloader):
                        data = {key: value.to(device) for key, value in batch.items()}
                        with torch.no_grad():
                            preds = model(data["code_ids"])
                            for ref in data["comments_ids"]:
                                ref = list(ref.cpu().numpy())
                                if padding_id in ref:
                                    ref = ref[:ref.index(padding_id)]
                                text = tokenizer.decode(
                                    ref[1:-1], clean_up_tokenization_spaces=False)
                                references.append([text])
                            for pred in preds:
                                t = pred[0].cpu().numpy()
                                t = list(t)
                                if padding_id in t:
                                    t = t[:t.index(padding_id)]
                                candidate = tokenizer.decode(
                                    t, clean_up_tokenization_spaces=False)
                                candidates.append([candidate])

                    model.train()
                    dict_size = len(candidates)
                    predictionMap = dict(zip(range(dict_size), candidates))
                    refMap = dict(zip(range(dict_size), references))
                    bleu_score = bleu.bleuFromMaps(refMap, predictionMap)
                    dev_bleu = round(bleu_score[0], 2)
                    print(bleu_score)
                    
                    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                    logger.info("  "+"*"*20)
                    if dev_bleu > best_bleu:
                        logger.info("  Best bleu:%s", dev_bleu)
                        logger.info("  "+"*"*20)
                        best_bleu = dev_bleu
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(
                            args.output_dir, "checkpoint-best-bleu")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, "module") else model  # Only save the model it-self
                        output_model_file = os.path.join(
                            output_dir, "model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        logger.info("  This epoch is not Best bleu:%s", round(best_bleu, 5))
                        logger.info("  "+"*"*20)


if __name__ == "__main__":
    main()