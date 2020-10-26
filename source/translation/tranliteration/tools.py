# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
from source.translation.tranliteration.data_helper \
    import create_or_get_voca, sentence_to_token_ids
from source.translation.tranliteration.model import Encoder, Decoder,  Transformer, greedy_decoder, Greedy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')


class Transliteration(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, x_test_path=None, result_file=None, batch_size=1):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = self.checkpoint['seq_len']
        self.en_voc = create_or_get_voca(vocabulary_path=dictionary_path + "vocab40.en")
        self.ko_voc = create_or_get_voca(vocabulary_path=dictionary_path + "vocab1000.ko")
        self.x_test_path = x_test_path
        self.result_file = result_file
        self.batch_size = batch_size
        self.EOS_ID = self.ko_voc['</s>']  # 1 End Token
        self.lines = []
        self.greedy = None
        self.model = self.model_load()

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Transformer(encoder, decoder)
        model = nn.DataParallel(model)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        self.greedy = Greedy(model=model, seq_len=self.seq_len)
        return model

    def src_input(self, sentence):
        idx_list = sentence_to_token_ids(sentence, self.en_voc)
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def tensor2sentence(self, indices: torch.Tensor) -> list:
        translation_sentence = []
        vocab = {v: k for k, v in self.ko_voc.items()}
        for idx in indices:
            if idx != 1:
                translation_sentence.append(vocab[idx])
            else:
                break
        translation_sentence = ''.join(translation_sentence).strip()
        while translation_sentence.find('<unk>') != -1:
            translation_sentence = translation_sentence.replace("<unk>", " ")
        return translation_sentence

    def transform(self, sentence: str) -> (str, torch.Tensor):
        enc_input = self.src_input(sentence)
        greedy_dec_input = self.greedy.greedy_decoder(enc_input)
        output, _ = self.model(enc_input, greedy_dec_input)
        indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
        output_sentence = self.tensor2sentence(indices)
        return output_sentence
