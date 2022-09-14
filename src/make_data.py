import os

import csv
import json
import random
import argparse

from collections import Counter
from multiprocessing import Pool
import multiprocessing as multi

import tqdm

def load_args():
    parser = argparse.ArgumentParser(description="janomeとbpeでトークナイズしたデータから語彙をカウントし、語彙idに置き換え")
    parser.add_argument("data_path", help="CirrusDumpのパス")
    parser.add_argument("--output_dir", default="data", help="書き出し先")
    parser.add_argument("--vocab_path", default="data/vocab.json", help="語彙ファイル保存先")
    parser.add_argument("--fairseq_vocab_path", default="data/dict.txt", help="fairseq用語彙ファイル保存先")
    parser.add_argument("--dev_size", default=0.01, help="開発データのサイズ")
    parser.add_argument("--test_size", default=0.01, help="テストデータのサイズ")
    parser.add_argument("--vocab_size", default=24000, help="語彙サイズ")
    parser.add_argument("--seed", default=1234, help="ランダムシード")
    return parser.parse_args()

def count_vocab(data):
    vocab = Counter()
    for d in data:
        d = d.strip()
        tokens = d.split(" ")
        vocab += Counter(tokens)
    return vocab

def map_vocab(inputs):
    data, vocab = inputs

    new_data = []
    for d in data:
        d = d.strip()
        tokens = d.split(" ")

        token_ids = [vocab[t] for t in tokens]

        new_data.append(" ".join(map(str, token_ids)))
    return new_data

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

def split_data(args, data):
    random.seed(args.seed)
    data_ids = list(range(len(data)))
    random.shuffle(data_ids)

    dev_size = int(len(data) * args.dev_size)
    dev_data = [data[i] for i in data_ids[:dev_size]]

    test_size = int(len(data) * args.test_size)
    test_data = [data[i] for i in data_ids[dev_size: dev_size+test_size]]

    train_data = [data[i] for i in data_ids[dev_size+test_size:]]

    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print(f"Test size: {len(test_data)}")
    return train_data, dev_data, test_data

def save_data(file_path, data):
    with open(file_path, "w") as f:
        f.write("\n".join(data))

def save_fairseq_vocab(file_path, table):
    with open(file_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter = ' ')
        writer.writerows(table)

def main():
    args = load_args()

    with open(args.data_path, "r") as f:
        raw_data = f.readlines()

    tasks = []
    n = 2000
    for i, span in enumerate(range(0, len(raw_data), n)):
        sub_data = raw_data[span: span+n]
        tasks.append(sub_data)

    vocab = Counter()
    with Pool(multi.cpu_count()-1) as p, \
        tqdm.tqdm(total=len(tasks)) as t:
        for _vocab in p.imap(count_vocab, tasks):
            t.update()
            vocab += _vocab

    most_tokens, most_num_tokens = map(list, zip(*vocab.most_common()))
    token2id = dict(zip(most_tokens, range(len(most_tokens))))

    tasks = [(task, token2id) for task in tasks]
    data = []
    with Pool(multi.cpu_count()-1) as p, \
        tqdm.tqdm(total=len(tasks)) as t:
        for text in p.imap(map_vocab, tasks):
            t.update()
            data += text

    fairseq_vocab = [*zip(range(len(most_num_tokens)), most_num_tokens)][:args.vocab_size]
    save_fairseq_vocab(args.fairseq_vocab_path, fairseq_vocab)

    roberta_vocab = ["<s>", "<pad>", "</s>", "<unk>"] + most_tokens[:args.vocab_size] + ["<mask>"]
    roberta_vocab = dict(zip(roberta_vocab, range(len(roberta_vocab))))

    save_json(args.vocab_path, roberta_vocab)

    train, dev, test = split_data(args, data)

    save_data(os.path.join(args.output_dir, "train.txt"), train)
    save_data(os.path.join(args.output_dir, "dev.txt"), dev)
    save_data(os.path.join(args.output_dir, "test.txt"), test)


if __name__ == "__main__":
    main()
