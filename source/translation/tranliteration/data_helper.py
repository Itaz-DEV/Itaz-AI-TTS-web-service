import os
import torch
import shutil
from abc import *
import sentencepiece as spm
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_BOS = "<s>"
_EOS = "</s>"
_UNK = "<unk>"
_PAD = "<pad>"
_START_VOCAB = [_BOS, _EOS, _UNK, _PAD]

BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3


def basic_tokenizer(sentence):
    return list(sentence.lower().strip())


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, encoding='utf-8') as data:
        lines = data.readlines()
        counter = 0
        for i in lines:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = tokenizer(i) if tokenizer else basic_tokenizer(i)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
    f = open(vocabulary_path, 'w', encoding='UTF8')
    for i in vocab_list:
        f.write(i + "\n")
    f.close()


def create_or_get_voca(vocabulary_path):
    with open(vocabulary_path, 'r', encoding='utf-8') as data:
        rev_vocab = [line.strip() for line in data.readlines()]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


class TranslationDataset(Dataset, metaclass=ABCMeta):
    # Translation Dataset abstract class
    # download, read data 등등을 하는 파트.
    def __init__(self, x_path, y_path, en_voc, ko_voc, sequence_size):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.y = open(y_path, 'r', encoding='utf-8').readlines()  # English data 위치
        self.ko_voc = ko_voc  # Korean 사전
        self.en_voc = en_voc  # English 사전
        self.sequence_size = sequence_size  # sequence 최대길이

    def __len__(self):  # data size를 넘겨주는 파트
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    @abstractmethod
    def encoder_input_to_vector(self, sentence: str): pass

    @abstractmethod
    def decoder_input_to_vector(self, sentence: str): pass

    @abstractmethod
    def decoder_output_to_vector(self, sentence: str): pass


class LSTMSeq2SeqDataset(TranslationDataset):
    # for Seq2Seq model & Seq2Seq attention model
    # using google sentencepiece (https://github.com/google/sentencepiece.git)
    def __init__(self, x_path, y_path, en_voc, ko_voc, sequence_size):
        super().__init__(x_path, y_path, en_voc, ko_voc,  sequence_size)
        self.KO_PAD_ID = ko_voc['<pad>']  # 3 Padding
        self.EN_PAD_ID = en_voc['<pad>']  # 3 Padding
        self.EN_BOS_ID = en_voc['<s>']  # 0 Start Token
        self.EN_EOS_ID = en_voc['</s>']  # 1 End Token

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str):
        idx_list = sentence_to_token_ids(sentence, self.en_voc)
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        idx_list = sentence_to_token_ids(sentence, self.ko_voc)
        idx_list.insert(0, self.EN_BOS_ID)  # Start Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str):
        idx_list = sentence_to_token_ids(sentence, self.ko_voc)
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list


class LSTMTestDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, x_path, ko_voc=None, sequence_size=20):
        self.KO_PAD_ID = ko_voc['<pad>']  # 3 Padding
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.ko_voc = ko_voc  # Korean 사전
        self.sequence_size = sequence_size  # sequence 최대길이

    def __len__(self):  # data size를 넘겨주는 파트
        return len(self.x)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        return encoder_input

    def encoder_input_to_vector(self, sentence: str):
        idx_list = sentence_to_token_ids(sentence, self.en_voc)
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list)

    def decoder_input_to_vector(self, sentence: str):
        return None

    def decoder_output_to_vector(self, sentence: str):
        return None

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list
