import sys
sys.path.append('..')

import pandas as pd
import argparse
from tqdm import tqdm
from utils.preprocess import TextTokenizer

parser = argparse.ArgumentParser('Clean csv data')
parser.add_argument('-i', '--csv_in', type=str, help='csv in')    
parser.add_argument('-o', '--csv_out', default=None, type=str, help='csv out')    
args = parser.parse_args()


def clean(args):
    df = pd.read_csv(args.csv_in)

    if args.csv_out is None:
        args.csv_out = args.csv_in[:-4]+ '_clean.csv'

    tokenizer = TextTokenizer(
        preprocess_steps=['base', 'replace_consecutive']
    )
    texts = df.title

    clean_tokens = []

    for text in tqdm(texts):
        tokens = tokenizer.tokenize(text)
        txt = ' '.join(tokens)
        clean_tokens.append(txt)

    df['cleaned_title'] = clean_tokens

    df.to_csv(args.csv_out, index=False)

if __name__ == '__main__':
    clean(args)

