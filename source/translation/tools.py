# -*- coding:utf-8 -*-
import re
import time
import torch
import torch.nn as nn

from nltk.tokenize import WordPunctTokenizer
from source.translation.utils import PostProcessing
from source.translation.data_helper import create_or_get_voca
from source.translation.model import Encoder, Decoder, Transformer, Beam, Greedy
from source.translation.tranliteration.tools import Transliteration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def change_hangle(token):
    """
    자음 또는 모음 중 sentencepiece에 들어가면서 ASCII코드가 바뀐경우를 원래대로
    :param token: 자음 또는 모음
    :return:
    """
    before = ['ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ', 'ᄊ', 'ᄋ',  'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ',
              'ᄒ', 'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ',
              'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ']
    after = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ',
             'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ',  'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ',
             'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    result = ""
    token_list = list(token)
    for i in token_list:
        if i in before:
            idx = before.index(i)
            result += after[idx]
        else:
            result += i
    return result


def isHangle(sentence):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', sentence))
    return hanCount > 0


def split_sentence(sentence):
    start = time.time()
    tf = [False, False, False]      # 한글, 영어, 특수문자
    if ord('가') <= ord(sentence[0]) <= ord('힣'):
        tf[0] = True
    elif ord('a') <= ord(sentence[0].lower()) <= ord('z'):
        tf[1] = True
    else:
        tf[2] = True

    lst = [sentence[0]]
    for i in sentence[1:]:
        if ord('가') <= ord(i) <= ord('힣'):
            if tf[0]:
                lst.append(i)
                tf = [True, False, False]
            else:
                lst.append(" ")
                lst.append(i)
                tf = [True, False, False]
        elif ord('a') <= ord(i.lower()) <= ord('z'):
            if tf[1]:
                lst.append(i)
                tf = [False, True, False]
            else:
                lst.append(" ")
                lst.append(i)
                tf = [False, True, False]

        elif i == " ":
            lst.append(i)
            tf = [True, True, True]
        else:
            if tf[2]:
                lst.append(i)
                tf = [False, False, True]
            else:
                lst.append(" ")
                lst.append(i)
                tf = [False, False, True]
    print('Split sentence time: {}'.format(time.time() - start))
    return "".join(lst)


class Translation(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, region=None, beam_search=False, k=3):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = 200
        self.beam_search = beam_search
        self.k = k
        self.ko_voc, self.en_voc = create_or_get_voca(save_path=dictionary_path,
                                                      ko_vocab_size=self.checkpoint['encoder_parameter']['input_dim'],
                                                      di_vocab_size=self.checkpoint['decoder_parameter']['input_dim'],
                                                      region=region)
        self.EOS_ID = self.ko_voc['</s>']  # 1 End Token
        self.processing = PostProcessing()
        self.literation = Transliteration(checkpoint='source/translation/Model/transliteration/best_seq2seq.pth',
                                          dictionary_path='source/translation/Dictionary/transliteration/')

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Transformer(encoder, decoder)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        if self.beam_search:
            self.beam = Beam(model=model, beam_size=self.k, seq_len=self.seq_len)
            self.greedy = Greedy(model=model, seq_len=self.seq_len)
        else:
            self.greedy = Greedy(model=model, seq_len=self.seq_len)
        return model

    def src_input(self, sentence):
        idx_list = self.ko_voc.EncodeAsIds(sentence)
        idx_list.append(self.EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def korean2dialect(self, model, sentence: str) -> (str, torch.Tensor):
        sentence = re.sub('[()]', ' ', sentence)
        sentence = ' '.join(sentence.split())
        if not isHangle(sentence):
            if str.isdigit(sentence):
                return self.transliteration(re.sub('[,]', '', sentence))
            else:
                return self.transliteration(sentence)
        enc_input = self.src_input(sentence)
        if self.beam_search:
            self.beam.beam_initialize()
            beam_dec_input = self.beam.beam_search_decoder(enc_input).unsqueeze(0)
            output, _ = model(enc_input, beam_dec_input)
        else:
            greedy_dec_input = self.greedy.greedy_decoder(enc_input)
            output, _ = model(enc_input, greedy_dec_input)
        indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
        output_sentence = self.tensor2sentence_di(indices)[0]
        output_sentence = self.processing.post_processing(sentence, output_sentence)
        if "<un>" in output_sentence:      # Unk 발생시 원래문장 출력
            return self.transliteration(sentence)
        elif sentence.count("<") != output_sentence.count("<"):
            return self.transliteration(sentence)

        output_sentence = self.transliteration(output_sentence)
        return output_sentence

    def tensor2sentence_di(self, indices: torch.Tensor) -> list:
        result = []
        translation_sentence = []
        temp = [indices[0]]
        for i in indices[1:]:
            # 2는 Unk 4는 띄어쓰기
            if i not in [2, 4]:
                temp.append(i)
            elif (i == 2 and temp[-1] == 2) or (i == 4 and temp[-1] == 4):
                continue
            elif len(temp) >= 2 and (i in [2, 4] and temp[-1] == 4 and temp[-2] == 2):
                # unk unk 또는 unk  일 경우 제거
                continue
            else:
                temp.append(i)

        indices = temp

        for idx in indices:
            word = self.en_voc.IdToPiece(idx)
            word = change_hangle(word)
            if word == '</s>':  # End token 나오면 stop
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()  # sentencepiece 에 _ 제거
        translation_sentence = ''.join(translation_sentence).replace('  ', ' ').strip()  # sentencepiece 에 _ 제거
        result.append(translation_sentence)
        return result

    def transliteration(self, sentence):
        def is_english(s):  # 영어 단어 포함이면 True, 영어 단어 미포함이면 False
            import string
            if any(char.lower() in string.ascii_lowercase for char in s):
                return True
            else:
                return False
        if not is_english(sentence):    # 문장에 영어단어가 없으면 그대로 리턴
            return sentence
        sentence = split_sentence(sentence)
        lst = WordPunctTokenizer().tokenize(sentence)
        for i in lst:
            if is_english(i):
                if any(map(str.isdigit, i)):
                    for j in i:
                        if is_english(j):
                            sentence = sentence.replace(j, self.literation.transform(j), 1)
                        else:
                            pass
                else:
                    sentence = sentence.replace(i, self.literation.transform(i), 1)
            else:
                pass
        return sentence
