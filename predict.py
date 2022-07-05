# coding=utf-8

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

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
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to model: Should contain the .bin files" )

    # Other parameters
    parser.add_argument("--max_seq", type=int,
                        default=256, help="maximum input sequence len")
    parser.add_argument("--max_output_seq", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--with_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--cuda_devices", type=str, nargs="+",
                        default="0", help="CUDA device ids")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--beam_size", default=5, type=int,
                        help="beam size for beam search")
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
    model.load_state_dict(torch.load(args.load_model_path))
    model.to(device)
    if args.n_gpu>1:
        model = nn.DataParallel(model,device_ids=args.cuda_devices)

    # Prepare data loader
    my_dir = os.path.join("data",args.dataset_name)
    code_path = os.path.join(my_dir,"test","code.txt")
    NL_path = os.path.join(my_dir,"test","nl.txt")
    if args.model_name_or_path == "microsoft/codebert-base":
        instance = get_instances(
            code_path, NL_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask = False, tokenizer=tokenizer, SOS = "<s>")
    else:
        instance = get_instances(code_path, NL_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask = False, tokenizer=tokenizer, SOS = "<mask0>")
    pred_dataset = BertDataset(instance)
    pred_dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)
    
    # Calculate bleu
    model.eval()
    logger.info("***** Running calculating bleu *****")
    candidates = []
    references = []
    pred_text = open(args.output_dir+"/result.txt","w",encoding="utf-8")
    idx = 0
    for batch in tqdm(pred_dataloader):
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
                idx = idx + 1
                t = pred[0].cpu().numpy()
                t = list(t)
                if padding_id in t:
                    t = t[:t.index(padding_id)]
                candidate = tokenizer.decode(
                    t, clean_up_tokenization_spaces=False)
                candidates.append([candidate])
                pred_text.write(str(idx)+ "\t" + candidate)


    model.train()
    dict_size = len(candidates)
    predictionMap = dict(zip(range(dict_size), candidates))
    refMap = dict(zip(range(dict_size), references))
    bleu_score = bleu.bleuFromMaps(refMap, predictionMap)
    dev_bleu = round(bleu_score[0], 2)
    print(bleu_score)
    
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  "+"*"*20)


if __name__ == "__main__":
    main()