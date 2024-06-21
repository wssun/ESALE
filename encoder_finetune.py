import argparse

from utils.bert_dataset import get_instances, BertDataset
import os
import torch
from torch.utils.data import DataLoader
from models.ESALE_model import ESALE
from models.trainers import Trainer
import logging
import os
from transformers import RobertaConfig,RobertaModel,RobertaTokenizer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None,required=True,
                        type=str, help="e.g. outputdir/ESALE") # required
    parser.add_argument("--model_name_or_path", default=None, type=str,required=True,
                        help="Path to pre-trained model: e.g. microsoft/unixcoder-base or microsoft/codebert-base")# required
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="The name of dataset. It should be JCSD or PCSD.")
    parser.add_argument("--hidden", type=int,
                        default=768, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int,
                        default=6, help="number of layers")
    parser.add_argument("--attn_heads", type=int,
                        default=12, help="number of attention heads")
    parser.add_argument("--max_seq", type=int,
                        default=256, help="maximum input sequence len")
    parser.add_argument("--max_output_seq", type=int,
                        default=128, help="maximum output sequence len")
    parser.add_argument("--awp_cls", type=int,
                        default=40, help="number of action words")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="number of batch_size")
    parser.add_argument("--epochs", type=int,
                        default=50, help="number of epochs")

    parser.add_argument("--with_test", action='store_true',
                        help="whether to test")
    parser.add_argument("--with_mlm", action='store_true',
                        help="whether to use masked language model")
    parser.add_argument("--with_ulm", action='store_true',
                        help="whether to use unidirectional language model")
    parser.add_argument("--with_awp", action='store_true',
                        help="whether to use action word prediction")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--with_cuda",action='store_true',
                        help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=str, nargs="+",
                        default="0", help="CUDA device ids")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, help="adam first beta value")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

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
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    if args.model_name_or_path == "microsoft/unixcoder-base":
        config.is_decoder = True
    # budild model
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    my_dir = os.path.join("data",args.dataset_name)
    instances = []
    for file_type in ["train","valid","test"]:
        code_path = os.path.join(my_dir, file_type, "code.txt")
        NL_path = os.path.join(my_dir, file_type, "nl.txt")
        AWP_path = os.path.join(my_dir, file_type, "AWP_"+str(args.awp_cls)+".txt")
        if args.model_name_or_path == "microsoft/codebert-base":
            instance = get_instances(
                code_path, NL_path, AWP_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask=True, tokenizer=tokenizer, SOS = "<s>")
        else:
            instance = get_instances(
                code_path, NL_path, AWP_path, max_seq=args.max_seq, max_output_seq = args.max_output_seq, is_mask=True, tokenizer=tokenizer, SOS = "<mask0>")   
        instances.append(instance)
    train_dataset = BertDataset(instances[0])
    test_dataset = BertDataset(instances[1])
    pred_dataset = BertDataset(instances[2])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    pred_dataloader = DataLoader(pred_dataset, batch_size=args.batch_size)

    model = ESALE(encoder=encoder, config=config, max_seq=args.max_seq, max_output_seq = args.max_output_seq,
                 awp_cls=args.awp_cls, with_awp=args.with_awp, with_ulm=args.with_ulm)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    model.to(device)

    print("Creating BERT Trainer")
    trainer = Trainer(model, train_dataloader=train_dataloader, test_dataloader=eval_dataloader, with_mlm=args.with_mlm,
                             with_ulm=args.with_ulm, with_awp=args.with_awp, lr=args.lr, betas=(
                                 args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                             with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, eps=args.adam_epsilon)

    print("Training Start")
    best_loss = 10000
    for epoch in range(args.epochs):
        trainer.train(epoch, False)
        if eval_dataloader is not None:
            loss = trainer.train(epoch, True)
        if best_loss > loss:
            best_loss = loss
            trainer.save(epoch, args.output_dir)


if __name__ == "__main__":
    main()
