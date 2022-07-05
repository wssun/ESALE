import argparse
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from meteor.meteor import Meteor

def get_M_and_R(hyp, ref, max_len):
    with open(hyp, "r") as r:
        hypothesis = r.readlines()
        res = {}
        for line in hypothesis:
            k = line.split("\t")[0]
            v = [" ".join(line.split("\t")[1].strip().lower().split()[:max_len])]
            res[k] = v
    
    with open(ref, "r") as r:
        references = r.readlines()
        gts = {} # ground truth
        for line in references:
            k = line.split("\t")[0]
            v = [" ".join(line.split("\t")[1].strip().lower().split()[:])]
            if k in res:
                gts[k] = v

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("ROUGe: "), score_Rouge

    return scores_Meteor, scores_Rouge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds",
        default=None,
        type=str,
        required=True,
        help="Path to predicted summaries.",
    )
    parser.add_argument(
        "--refs",
        default=None,
        type=str,
        required=True,
        help="Path to references.",
    )
    return parser.parse_args()

def main(args):
    hyp_path = args.preds
    ref_path = args.refs
    get_M_and_R(hyp_path, ref_path, 128)

if __name__ == "__main__":
    args = parse_args()
    main(args)