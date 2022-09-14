# fairseq_roberta_preprocessor
fairseqでrobertaを学習する場合に必要な前処理を施します。

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/tokenize_janome.py [cirrussearchのパス]
mkdir -p data
subword-nmt learn-bpe -s 10000 < cache/janome.txt > data/codes.txt
subword-nmt apply-bpe -c data/codes.txt < cache/janome.txt > cache/janome_bpe.txt
python3 src/make_data.py cache/janome_bpe.txt
fairseq-preprocess \
    --only-source \
    --srcdict data/dict.txt \
    --trainpref data/train.txt \
    --validpref data/dev.txt \
    --testpref data/test.txt \
    --destdir data/data-bin/ \
    --workers 26
~~~
