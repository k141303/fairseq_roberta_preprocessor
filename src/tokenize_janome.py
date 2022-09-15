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
    parser.add_argument("--output_dir", default="cache/janome", help="トークナイズデータの書き出しフォルダ")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def tokenize(inputs):
    data, output_path = inputs

    data_iter = iter(data)
    data = []
    for head, cont in zip(data_iter, data_iter):
        cont = json.loads(cont)
        data.append(cont["text"])

    t = Tokenizer(wakati=True)

    for i in range(len(data)):
        data[i] = data[i].splitlines()
        data[i] = " ".join(data[i])

        l_space = re.match("^(\s*)", data[i]).group(1)

        data[i] = list(t.tokenize(data[i]))
        if len(l_space) > 0:
            data[i] = [l_space] + list(data[i])

        data[i] = [re.sub("\s", " ▁ ", t) for t in data[i]]
        data[i] = " ".join(data[i])

    data = "\n".join(data) + "\n"

    with open(output_path, "w") as f:
        f.write(data)

def main():
    args = load_args()

    raw_data = []
    with gzip.open(args.cirrus_path, mode="rb") as f, \
        tqdm.tqdm() as t:
        for line in f:
            raw_data.append(line)
            t.update()
            if args.debug and len(raw_data) >= 20000:
                break

    tasks = []
    n = 2000
    for i, span in enumerate(range(0, len(raw_data), n)):
        sub_data = raw_data[span: span+n]
        output_path = os.path.join(args.output_dir, f"{i}.txt")
        tasks.append((sub_data, output_path))

    os.makedirs(args.output_dir, exist_ok=True)
    with Pool(multi.cpu_count()-1) as p, \
        tqdm.tqdm(total=len(tasks)) as t:
        for _ in p.imap(tokenize, tasks):
            t.update()

if __name__ == "__main__":
    main()
