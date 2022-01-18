# coding=utf8
import pandas as pd
import re

TOKENS = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']


class Vocab:
    """vocab for moses smiles"""
    pattern = '(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    token_regex = re.compile(pattern)
    def __init__(self, trainset):
        """init vocab
        Args:
            trainset: trainset filename
        """
        self.token_list = self.get_tokens_from_corpus(trainset)
        self.t2i = {t:i for i, t in enumerate(self.token_list)}
        self.i2t = {i:t for i, t in enumerate(self.token_list)}

    def get_size(self):
        return len(self.token_list)

    def get_tokens_from_corpus(self, trainset):
        """get tokens from corpus"""
        df = pd.read_csv(trainset)
        tokens = set([])
        for smiles in df['SMILES']:
            tokens.update(self.tokenize_smiles(smiles))
        return TOKENS + list(tokens)

    def tokenize_smiles(self, smiles):
        """tokenize smiles"""
        return Vocab.token_regex.findall(smiles)


    def translate_sentence(self, sentence:str):
        """convert sentence to index
        Args:
            sentence: smiles (str type)
        Return:
            indexes: list of index
        """
        indexes = [self.t2i[t] if t in self.t2i else self.t2i['<UNK>'] for t in sentence]
        return indexes

    def translate_index(self, indexes):
        """convert indexes to smiles
        Args:
            indexes: list of index
        Returns:
            sentence: smiles(str)
        """
        sentence = [self.i2t[i] for i in indexes]
        sentence = ''.join(sentence)
        return sentence

    def append_delimiters(self, sentence):
        """add <SOS> and <EOS>"""
        return ['<SOS>'] + list(sentence) + ['<EOS>']

    @property
    def SOS(self):
        return self.t2i['<SOS>']

    @property
    def EOS(self):
        return self.t2i['<EOS>']

    @property
    def PAD(self):
        return self.t2i['<PAD>']

    @property
    def UNK(self):
        return self.t2i['<UNK>']
