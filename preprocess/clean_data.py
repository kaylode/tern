import sys
sys.path.append('..')

import pandas as pd
import argparse
from tqdm import tqdm
from utils.preprocess import TextTokenizer
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser('Clean csv data')
parser.add_argument('-i', '--csv_in', type=str, help='csv in')    
parser.add_argument('-o', '--csv_out', default=None, type=str, help='csv out')    
args = parser.parse_args()


def encode_labels(df):
    lbl_encoder = LabelEncoder()
    df['label_code'] = lbl_encoder.fit_transform(df['label_group'])
    return df

def set_target(df):
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
    return df

def remove_duplicated_phashs(df_in):
    df_out = df_in.drop_duplicates(['image_phash','cleaned_title'], keep='last')
    return df_out

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

    # Remove duplicates if both have same phash and title
    before = len(df)
    df = remove_duplicated_phashs(df)
    print(f"Removed {len(df) - before} duplicated phashs")

    # Set target group for each item
    df = set_target(df)
    df = encode_labels(df)

    df.to_csv(args.csv_out, index=False)

if __name__ == '__main__':
    clean(args)

