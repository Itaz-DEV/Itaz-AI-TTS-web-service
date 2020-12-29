# -*- coding: cp949 -*-
import time
from source.translation.tools import Translation
import torch
region='gyeong'

def count_space(sentence):
	return sentence.count(' ')


translation = Translation(checkpoint='source/translation/Model/' + str(region) + '/best_transformer.pth',
						  dictionary_path='source/translation/Dictionary/' + str(region),
									   beam_search=None, k=0, region=region)
model = translation.model_load()

def transform(sentence):
	start = time.time()
	sentence = ' '.join(sentence.split())
	token = sentence.split(' ')
	tmp = token[0]
	lst_gy = []
	with torch.no_grad():
		if len(token) == 1:
			if count_space(tmp) > 0:
				lst_gy.append(translation.korean2dialect(model, tmp))
			else:
				lst_gy.append(tmp)
		else:
			for j in range(1, len(token)):
				tmp = ' '.join([tmp, token[j]])
				if len(tmp) >= 35:
					if count_space(tmp) > len(tmp) * 0.1:
						lst_gy.append(translation.korean2dialect(model, tmp))
						tmp = ''
					else:
						lst_gy.append(tmp)
						tmp = ''
				elif j == len(token)-1:
					lst_gy.append(translation.korean2dialect(model, tmp))
	output_sentence = ' '.join(lst_gy)
	return output_sentence



import pandas as pd
standard_text = pd.read_excel('standard_gyoungsang.xlsx')
gyeongsang_lst=[]
for row in standard_text.itertuples():
	print(row[0])
	gyeongsang_lst.append(transform(row[1]))

df = pd.DataFrame(gyeongsang_lst)
df.to_csv(f'{region}.csv',encoding='cp949')
