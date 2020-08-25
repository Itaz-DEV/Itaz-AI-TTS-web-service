# -*- coding: cp949 -*-
import sys
import json
import time
import torch
from source.db.config import DB_URL
from flask_sqlalchemy import SQLAlchemy
from source.db.app import TTS

sys.path.append('source/waveglow/')
from flask import Flask, render_template, request
from translation import Korean2Dialect
from speech_synthesis import Text2Speech
from koalanlp import API
from koalanlp.proc import SentenceSplitter
from koalanlp.Util import initialize, finalize
#### split paragraph to list of sentences
initialize(hnn="2.1.3")

def clean_text(txt:str)->list:
    start_time= time.time()
    ### transform english char to korean text
    transform_dict = {'a':'����','b':'��','c':'��','d':'��','e':'��','f':'����','g':'��','h':'����ġ','i':'����','j':'����','k':'����','l':'��','m':'��',
                      'n':'��','o':'��','p':'��', 'q':'ť','r':'�Ƹ�','s':'����','t':'Ƽ','u':'��','v':'����','w':'������','x':'����','y':'����','z':'��Ʈ',
                      u"'":u'"', '(':', ', ')':', ', '#':'��', '%':'����', '@':'������', '+':'���ϱ�', '-':'����', ':':'������', '*':'��'}
    ### remove not allowed chars
    not_allowed_characters = list('^~')
    txt = ''.join(i for i in txt if not i in not_allowed_characters)
    txt = txt.lower().strip()
    ### transform special char to hangul
    for k,v in transform_dict.items():
        txt=txt.replace(k, v).replace(' .', '.').replace(' ?', '?').strip()
    splitter = SentenceSplitter(api=API.HNN)
    paragraph = splitter(txt)
    # return paragraph
    txt_list=[]
    import string
    max_len=50
    for s in paragraph:
        txt_ = s.translate(str.maketrans('', '', string.punctuation.replace(',','').replace('.','')))
        txt_=txt_.strip()

        while True:
            if ',,' in txt_:
                txt_=txt_.replace(',,',',')
            else:
                break

        while True:
            if '..' in txt_:
                txt_=txt_.replace('..','.')
            else:
                break

        if len(txt_.replace(',','').replace(' ','').strip())>0:
            txt_ = txt_.replace(' ,', ',').replace(',', ', ')
            if len(txt_) >= max_len:
                start = 0
                while True:
                    if start>=len(txt_):
                        break
                    else:
                        sub_txt = txt_[start:start+max_len]
                        start += max_len
                        if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
                            sub_txt = sub_txt + '.'
                        txt_list.append(sub_txt.strip())
            else:
                if not (txt_.endswith('.') or txt_.endswith('?') or txt_.endswith('!')):
                    txt_ = txt_ + '.'
                txt_list.append(txt_.strip())
    print('Cleaning Text time: {}'.format(time.time() - start_time))
    return txt_list


# =============================== define web app =======================================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']= DB_URL
db = SQLAlchemy(app)

jeju_translation = Korean2Dialect('jeju', beam_search=False, k=0)           # ���� ���� Ŭ���� ����
gyeong_translation = Korean2Dialect('gyeong', beam_search=False, k=0)       # ��� ���� Ŭ���� ����
jeon_translation = Korean2Dialect('jeon', beam_search=False, k=0)           # ���� ���� Ŭ���� ����

jeju_speech = Text2Speech('����')  # ���� �����ռ� Ŭ���� ����
gyeong_speech = Text2Speech('���')  # ��� �����ռ� Ŭ���� ����
jeon_speech = Text2Speech('����')                                            # ���� �����ռ� Ŭ���� ����


@app.route('/ml-inference', methods=['POST'])
def ml_inference():
    total_time = time.time()
    print('====== Synthesizing ======')
    print(request.form)

    gender = int(request.form['gender'])         # [0, 1] == ['����', '����']
    model_type = int(request.form['model'])      # [0, 1, 2] = ['���ֵ�', '���', '����]
    # ������ ����
    if model_type == 0:             # ���ֵ�
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type == 1:           # ���
        korean2dialect = gyeong_translation
        text2speech = gyeong_speech
    elif model_type == 2:           # ����
        korean2dialect = jeon_translation
        text2speech = jeon_speech
    else:
        print(model_type)
        raise NotImplementedError

    korean = request.form['input-text']  # ǥ�ؾ� Input
    dialect = korean2dialect.transform(korean)

    translated_length = len(korean) + int(len(korean) * 0.26)

    dialect = dialect[:translated_length] if len(dialect) > translated_length else dialect  # ����
    print(f'translated text: {dialect}')
    start=time.time()
    txt_list = clean_text(txt=dialect)  # ������ �ؽ�Ʈ Ŭ����
    print(f'cleaned text: {txt_list}')
    txt_list = txt_list[:4] if len(''.join(txt_list)) > translated_length else txt_list  # ����

    try:
        wav_file, error_log = text2speech.forward(txt_list)  # �ؽ�Ʈ -> wav file
        error_sentences = []
        for k, v in error_log.items():
            if v is True:
                error_sentences.append(k)
        error_sentences = '|'.join(error_sentences)
        return_data = {'translated_text': dialect, 'audio_stream': wav_file}
        res = app.response_class(response=json.dumps(return_data), status=200, mimetype='application/json')
        ip = request.remote_addr
        print('Total time(translation + synthesize): {}'.format(time.time() - total_time))
        tts = TTS(dialect_type=model_type, korean=korean, dialect=dialect, ip=ip, error=error_sentences)
        db.session.add(tts)
        db.session.commit()
        dialect, txt_list, wav_file, error_sentences, return_data, korean2dialect, text2speech = None, None, None, None, None, None, None
        torch.cuda.empty_cache()
        return res
    except Exception as e:
        print(e)
        dialect, txt_list, wav_file, error_sentences, return_data, korean2dialect, text2speech = None, None, None, None, None, None, None
        res = app.response_class(response=None, status=500, mimetype='application/json')
        torch.cuda.empty_cache()
        return res

@app.route('/api-inference', methods=['POST'])
def api_inference():
    total_time = time.time()
    print('====== Synthesizing ======')
    data = request.get_json()
    print(data)
    gender = int(data['gender'])         # [0, 1] == ['����', '����']
    model_type = int(data['model'])      # [0, 1, 2] = ['���ֵ�', '���', '����]
    # ������ ����
    if model_type == 0:             # ���ֵ�
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type == 1:           # ���
        korean2dialect = gyeong_translation
        text2speech = gyeong_speech
    elif model_type == 2:           # ����
        korean2dialect = jeon_translation
        text2speech = jeon_speech
    else:
        print(model_type)
        raise NotImplementedError

    korean = data['input-text']  # ǥ�ؾ� Input
    dialect = korean2dialect.transform(korean)
    translated_length = len(korean) + int(len(korean) * 0.26)

    dialect = dialect[:translated_length] if len(dialect)> translated_length else dialect  # ����
    print(f'translated text: {dialect}')
    txt_list = clean_text(txt=dialect)  # ������ �ؽ�Ʈ Ŭ����
    print(f'cleaned text: {txt_list}')
    txt_list = txt_list[:4] if len(''.join(txt_list)) > translated_length else txt_list  # ����

    # try:
    wav_file, error_log = text2speech.forward(txt_list)  # �ؽ�Ʈ -> wav file
    error_sentences = []
    for k, v in error_log.items():
        if v is True:
            error_sentences.append(k)
    error_sentences = '|'.join(error_sentences)
    return_data = {'translated_text': dialect, 'audio_stream': wav_file}
    res = app.response_class(response=json.dumps(return_data), status=200, mimetype='application/json')
    ip = request.remote_addr
    print('Total time(translation + synthesize): {}'.format(time.time() - total_time))
    tts = TTS(dialect_type=model_type, korean=korean, dialect=dialect, ip=ip, error=error_sentences)
    db.session.add(tts)
    db.session.commit()
    dialect, txt_list, wav_file, error_sentences, return_data, korean2dialect, text2speech = None, None, None, None, None, None, None
    torch.cuda.empty_cache()
    return res

@app.route('/')
def index():
    return render_template('index.html')
