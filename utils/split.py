import argparse
import os
import tqdm
from nltk.stem.porter import PorterStemmer


def get_total_nl(file_name):
    tgt = open(os.path.join("data", file_name, "total_nl.txt"), "w", encoding="utf-8")
    for file_type in ["train", "valid", "test"]:
        src = open(
            os.path.join("data", file_name, file_type, "nl.txt"), "r", encoding="utf-8"
        )
        for line in src.readlines():
            tgt.write(line)


def get_top_N_action_words(src, tgt, N):
    total = 0
    fwcnt = {}
    fwmap = {}
    source = open(src, "r", encoding="utf-8")
    target = open(tgt, "w", encoding="utf-8")
    porter_stemmer = PorterStemmer()
    lines = source.readlines()
    raw_size = len(lines)
    for line in lines:
        fw_in_line = line.split(" ")[0]
        fw_in_line = porter_stemmer.stem(fw_in_line)
        if fw_in_line not in fwcnt.keys():
            fwcnt[fw_in_line] = 1
        else:
            fwcnt[fw_in_line] += 1
    for i in range(N):
        max_fw = max(fwcnt, key=fwcnt.get)
        fwmap[max_fw] = str(i)
        target.write("%s, %s\n" % (max_fw, fwcnt[max_fw]))
        total += fwcnt[max_fw]
        del [fwcnt[max_fw]]
    fwmap["other"] = str(N)
    print(str(total) + " accounts for " + str(total / raw_size))
    return fwmap


def get_action_words(dataset, kind, fwmap, N):
    source_dir = os.path.join(dataset, kind, "nl.txt")
    dest_dir = os.path.join(dataset, kind, "AWP_" + str(N) + ".txt")
    source = open(source_dir, "r", encoding="utf-8")
    dest = open(dest_dir, "w", encoding="utf-8")
    porter_stemmer = PorterStemmer()
    for line in tqdm.tqdm(source.readlines()):
        fw_in_line = line.split(" ")[0]
        fw_in_line = porter_stemmer.stem(fw_in_line)
        if fw_in_line not in fwmap.keys():
            dest.write(str(N))  # means "other"
        else:
            dest.write(str(fwmap[fw_in_line]))
        dest.write("\n")


def main(args):
    get_total_nl(args.dataset_name)
    fwmap = get_top_N_action_words(
        os.path.join("data", args.dataset_name, "total_nl.txt"),
        os.path.join("data", args.dataset_name, str(args.aw_cls) + "_action_words.txt"),
        args.aw_cls,
    )
    for file_type in ["train", "valid", "test"]:
        get_action_words(
            os.path.join(
                "data",
                args.dataset_name,
            ),
            file_type,
            fwmap,
            args.aw_cls,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="The name of dataset. It should be JCSD or PCSD.",
    )
    parser.add_argument(
        "--aw_cls",
        default=None,
        type=int,
        required=True,
        help="The number of top N action words you want to count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
