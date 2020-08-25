import time
import torch
from source.translation.tools import Translation


def count_space(sentence):
    return sentence.count(' ')


class Korean2Dialect(object):
    def __init__(self, region, beam_search=False, k=0):
        self.translation = Translation(checkpoint='source/translation/Model/' + str(region) + '/best_transformer.pth',
                                       dictionary_path='source/translation/Dictionary/' + str(region),
                                       beam_search=beam_search, k=k, region=region)
        self.model = self.translation.model_load()

    def transform(self, sentence):
        start = time.time()
        token = sentence.split(' ')
        tmp = token[0]
        lst_gy = []
        with torch.no_grad():
            if len(token) == 1:
                if count_space(tmp) > 0:
                    lst_gy.append(self.translation.korean2dialect(self.model, tmp))
                else:
                    lst_gy.append(tmp)
            else:
                for j in token[1:]:
                    tmp = ' '.join([tmp, j])
                    if len(tmp) >= 35:
                        if count_space(tmp) > len(tmp) * 0.1:
                            lst_gy.append(self.translation.korean2dialect(self.model, tmp))
                            tmp = ''
                        else:
                            lst_gy.append(tmp)
                            tmp = ''
                    elif j == token[-1]:
                        lst_gy.append(self.translation.korean2dialect(self.model, tmp))
        output_sentence = ' '.join(lst_gy)
        print('Translation time: {}'.format(time.time() - start))
        return output_sentence