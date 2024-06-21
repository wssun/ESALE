import copy
from torch.utils.data import Dataset
import torch
from tqdm import trange
import random

MASK_RATE = 0.15


class TrainingInstance():
    def __init__(self, code_ids, comments_ids, masked_comments_ids, masked_ids, code_mask, comments_mask, awp_label):
        self.code_ids = code_ids
        self.comments_ids = comments_ids
        self.masked_comments_ids = masked_comments_ids
        self.masked_ids = masked_ids
        self.code_mask = code_mask
        self.comments_mask = comments_mask
        self.awp_label = awp_label


def pad_input(arr: list, max_seq, padding_id=1):
    if len(arr) > max_seq:
        arr = arr[:max_seq]
    if len(arr) < max_seq:
        padding = [padding_id]*(max_seq-len(arr))
        arr.extend(padding)
    return arr


def get_mask(input_ids: list or torch.Tensor, is_ulm=False, padding_id = 1):
    """ 
        maskT*mask and then get the lower triangular matrix
        input_ids: list
        return: tensor shape [len(list), len(list)]
    """
    if isinstance(input_ids, list):
        input_ids = torch.Tensor(input_ids)
    mask = (input_ids != padding_id).float()
    mask = mask.unsqueeze(dim=0)
    output_mask = torch.mm(torch.transpose(mask, 0, 1), mask)
    if is_ulm:
        output_mask = torch.tril(output_mask, 0)
    output_mask = -1e4*(1-output_mask)
    return output_mask


def tokenize_input_with_mask(list, idx, max_seq, tokenizer,padding_id):
    """
        output: 
        input_ids: tensor
        masked_inputs: tensor
        masked_ids: tensor
    """

    inputs = tokenizer.tokenize(list[idx])
    masked_inputs = copy.deepcopy(inputs)
    masked_ids = torch.zeros(max_seq)
    for i in range(min(len(inputs),max_seq-1)): # do not mask </s>
        rng = random.random()
        if rng < MASK_RATE:
            masked_ids[i] = 1
            masked_inputs[i] = "<mask>"
    inputs = inputs + ["</s>"]
    input_ids = tokenizer.convert_tokens_to_ids(inputs)
    input_ids = pad_input(input_ids, max_seq, padding_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    masked_inputs = masked_inputs + ["</s>"]
    masked_input_ids = tokenizer.convert_tokens_to_ids(masked_inputs)
    masked_input_ids = pad_input(masked_input_ids, max_seq, padding_id)
    masked_input_ids = torch.tensor(masked_input_ids, dtype=torch.long)
    masked_ids = torch.tensor(masked_ids, dtype=torch.int)
    return input_ids, masked_input_ids, masked_ids


def tokenize_input(list, idx, max_seq, tokenizer, padding_id, SOS = "<mask0>"):
    """output: tensor"""

    inputs = tokenizer.tokenize(list[idx])
    inputs = [SOS] + inputs + ["</s>"]
    input_ids = tokenizer.convert_tokens_to_ids(inputs)
    input_ids = pad_input(input_ids, max_seq, padding_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def get_instances(code_path, NL_path, AWP_path = None, max_seq = 256, max_output_seq = 128, is_mask=True, tokenizer=None, SOS = "<mask0>"):
    padding_id = tokenizer.convert_tokens_to_ids("<pad>")
    if AWP_path:
        awp_list = open(AWP_path, "r", encoding="utf-8").readlines()
    code_list = open(code_path, "r", encoding="utf-8").readlines()
    NL_list = open(NL_path, "r", encoding="utf-8").readlines()
    inst_num = len(code_list)
    instances = []
    for i in trange(inst_num):
        masked_comments_ids = None # comment list with <mask>
        awp_label = None
        masked_ids = None # ids of masked words
        code_ids = tokenize_input(code_list, i, max_seq, tokenizer, padding_id, SOS)
        if is_mask:
            comments_ids, masked_comments_ids, masked_ids = tokenize_input_with_mask(
                NL_list, i, max_output_seq, tokenizer, padding_id)
        else:
            comments_ids = tokenize_input(NL_list, i, max_output_seq, tokenizer,padding_id,SOS)
        code_mask = code_ids.ne(padding_id).long()
        comments_mask = get_mask(comments_ids, True)
        if AWP_path:
            awp_label = torch.tensor(int(awp_list[i]), dtype=torch.long)
            
        instance = TrainingInstance(code_ids, comments_ids, masked_comments_ids,
                                    masked_ids, code_mask, comments_mask, awp_label)
        instances.append(instance)
    return instances



class BertDataset(Dataset):

    def __init__(self, instances: TrainingInstance):
        self.instances = instances
        self.inst_nums = len(instances)

    def __len__(self):
        return self.inst_nums

    def __getitem__(self, index):
        instance = self.instances[index]
        output = {"code_ids": instance.code_ids,
                  "comments_ids": instance.comments_ids,
                  "masked_comments_ids": instance.masked_comments_ids,
                  "masked_ids": instance.masked_ids,
                  "code_mask": instance.code_mask,
                  "comments_mask": instance.comments_mask,
                  "awp_label": instance.awp_label,
                  }
        return {key: torch.tensor(value) for key, value in output.items() if value != None}


