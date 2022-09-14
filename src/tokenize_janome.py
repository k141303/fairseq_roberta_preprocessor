import os
import re

import gzip
import json

import argparse

import tqdm

from janome.tokenizer import Tokenizer

from multiprocessing import Pool
import multiprocessing as multi

def load_args():
    parser = argparse.ArgumentParser(description="CirrusDumpをjanomeでトークナイズ")
    parser.add_argument("cirrus_path", help="CirrusDumpのパス")
    parser.add_argument("--output_path", default="cache/janome.txt", help="CirrusDumpのパス")
    return parser.parse_args()

def tokenize(data):
    data_iter = iter(data)
    data = {}
    for head, cont in zip(data_iter, data_iter):
        head = json.loads(head)
        cont = json.loads(cont)
        data[head["index"]["_id"]] = cont["text"]

    t = Tokenizer(wakati=True)

    for key in data:
        data[key] = data[key].splitlines()
        data[key] = " ".join(data[key])

        l_space = re.match("^(\s*)", data[key]).group(1)

        data[key] = list(t.tokenize(data[key]))
        if len(l_space) > 0:
            data[key] = [l_space] + list(data[key])

        data[key] = [re.sub("\s", " ▁ ", t) for t in data[key]]
        data[key] = " ".join(data[key])

    data = [{"_id": k, "tokens": v} for k, v in data.items()]

    return [d["tokens"] for d in data]

def main():
    args = load_args()

    raw_data = []
    with gzip.open(args.cirrus_path, mode="rb") as f, \
        tqdm.tqdm() as t:

        cnt = 0
        for line in f:
            raw_data.append(line)
            t.update()

    tasks = []
    n = 2000
    for span in range(0, len(raw_data), n):
        sub_data = raw_data[span: span+n]
        # output_path = os.path.join(args.output_dir, f"{i}.json")
        tasks.append(sub_data)

    with Pool(multi.cpu_count()-1) as p, \
        tqdm.tqdm(total=len(tasks)) as t:
        for text in p.imap(tokenize, tasks):
            t.update()
            with open(args.output_path, "a") as f:
                f.write("\n".join(text) + "\n")

if __name__ == "__main__":
    main()
