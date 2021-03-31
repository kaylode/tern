import os
import random
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

os.environ["MODEL_DIR"] = './augmentations/augment_models'

class OneOf():
    def __init__(self, transforms):
        self.transforms = transforms
    def augment(self, text, n=1):
        func = random.choice(self.transforms)
        return func.augment(text, n=n)

class WithProb():
    def __init__(self, transform, p=1.0):
        self.transform = transform
        self.p = p
    def augment(self, text, n=1):
        prob = random.random()
        if prob>=self.p:
            return self.transform.augment(text, n = n)
        else:
            return text

class TextAugmentation():

    def __init__(self):
        self.transforms = [
            WithProb(naw.RandomWordAug(action='crop',aug_p=0.5, aug_min=0), p=0.5),      # Random crop word
            WithProb(nac.OcrAug(), p=0.3),            # OCR augment

            OneOf([
                # WithProb(naw.WordEmbsAug(
                #     model_type='glove', model_path=os.environ.get("MODEL_DIR")+'/glove.6B.300d.txt',
                #     action="substitute"), p=0.3),

                # WithProb(naw.ContextualWordEmbsAug(
                #     model_path='bert-base-uncased', 
                #     action="substitute"), p=0.3), # Random insert word by near embeddings
                # WithProb(naw.ContextualWordEmbsAug(
                #     model_path='distilbert-base-uncased', 
                #     action="substitute"), p=0.3), # Random subtitute word by near embeddings
                # WithProb(naw.ContextualWordEmbsAug(
                #     model_path='roberta-base', 
                #     action="substitute"), p=0.3), # Random subtitute word by near embeddings
                WithProb(naw.SynonymAug(aug_src='wordnet') , p=0.3),    # Replace symnonym words
            ]),

            # OneOf([
            # #     # WithProb(naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all'), p=0.3),  # Replace symnonym words
            # ]),
            
            # back_translation_aug = naw.BackTranslationAug(
            #     from_model_name='transformer.wmt19.en-de', 
            #     to_model_name='transformer.wmt19.de-en' 
            # )                                               # Backtranslation
        ]
        # nas.AbstSummAug(model_path='t5-base', num_beam=3)   # Abstractive summarization

    def __call__(self, text, n=1):
        for tf in self.transforms:
            text = tf.augment(text, n=n)
        return text

    