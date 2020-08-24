import time
from source.translation.tools import Translation


class Korean2Dialect(object):
    def __init__(self, region, beam_search=False, k=0):
        self.translation = Translation(checkpoint='source/translation/Model/' + str(region) + '/best_transformer.pth',
                                       dictionary_path='source/translation/Dictionary/' + str(region),
                                       beam_search=beam_search, k=k, region=region)
        self.model = self.translation.model_load()

    def transform(self, sentence):
        token = sentence.split(' ')
        tmp = token[0]
        lst_gy = []
        if len(token) ==1:
            lst_gy.append(self.translation.korean2dialect(self.model, tmp))
        else:
            for j in token[1:]:
                tmp = ' '.join([tmp, j])
                if len(tmp) >= 35:
                    lst_gy.append(self.translation.korean2dialect(self.model, tmp))
                    tmp = ''
                elif j == token[-1]:
                    lst_gy.append(self.translation.korean2dialect(self.model, tmp))
        # txt = self.translation.korean2dialect(self.model, sentence)
        return ' '.join(lst_gy)


if __name__ == '__main__':
    translation = Translation(checkpoint='source/translation/Model//gyeong/best_transformer.pth',
                              dictionary_path='source/translation//Dictionary/gyeong',
                              beam_search=True, k=5, region="gyeong")
    model = translation.model_load()
    start = time.time()
    output = translation.korean2dialect(model, "계좌번호를 알려주시면 바로 입금하고 확인해 드릴게요.")
    end = time.time() - start
    print("time: ", str(end))