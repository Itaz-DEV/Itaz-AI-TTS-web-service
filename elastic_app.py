# -*- coding: cp949 -*-
import sys
import json
import time
import torch

sys.path.append('source/waveglow/')
from flask import Flask, render_template, request
from translation import Korean2Dialect
from elastic_synthesis import Text2Speech

# from koalanlp import API
# from koalanlp.proc import SentenceSplitter
# from koalanlp.Util import initialize, finalize
#
# try:
#     #### split paragraph to list of sentences
#     initialize(hnn="2.1.3")
# except OSError:
#     pass

def clean_text(txt: str) -> list:
    start_time = time.time()
    # splitter = SentenceSplitter(api=API.HNN)
    # paragraph = splitter(txt)
    # return paragraph
    txt_list = []
    import string
    max_len = 60
    s=txt
    txt_ = s.translate(str.maketrans('', '', string.punctuation.replace(',', '').replace('.', '').replace('-', '').replace('/', '')))
    txt_ = txt_.strip()

    while True:
        if ',,' in txt_:
            txt_ = txt_.replace(',,', ',')
        else:
            break

    while True:
        if '..' in txt_:
            txt_ = txt_.replace('..', '.')
        else:
            break

    while True:
        if ',,' in txt_:
            txt_ = txt_.replace('--', '-')
        else:
            break

    while True:
        if '..' in txt_:
            txt_ = txt_.replace('//', '/')
        else:
            break

    if len(txt_.replace(',', '').replace(' ', '').strip()) > 0:
        if len(txt_) >= max_len:
            start = 0
            while True:
                if start >= len(txt_):
                    break
                else:
                    if len(txt_) >= start + max_len + 1:
                        while True:
                            if max_len>=50:
                                if txt_[start+max_len] ==' ' or txt_[start+max_len] =='?' or txt_[start+max_len] ==',' or txt_[start+max_len] =='.' or txt_[start+max_len] =='!':
                                    sub_txt = txt_[start:start + max_len]
                                    if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
                                        if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
                                            sub_txt = sub_txt + '.'
                                        txt_list.append(sub_txt.strip())

                                    start += max_len
                                    max_len=60
                                    break
                                else:
                                    max_len = max_len - 1
                            else:
                                sub_txt = txt_[start:start + max_len]
                                if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
                                    if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
                                        sub_txt = sub_txt + '.'
                                    txt_list.append(sub_txt.strip())

                                start += max_len
                                max_len = 60
                                break
                    else:
                        sub_txt = txt_[start:start + max_len]
                        start += max_len

                        if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
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

jeju_translation = Korean2Dialect('jeju', beam_search=False, k=0)  # 제주 번역 클래스 선언
gyeong_translation = Korean2Dialect('gyeong', beam_search=False, k=0)  # 경상 번역 클래스 선언
jeon_translation = Korean2Dialect('jeon', beam_search=False, k=0)  # 전라 번역 클래스 선언

jeju_speech = Text2Speech('제주')  # 제주 음성합성 클래스 선언
gyeong_speech = Text2Speech('경상')  # 경상 음성합성 클래스 선언
jeon_speech = Text2Speech('전라')  # 전라 음성합성 클래스 선언


def tts_inference(gender, model_type, korean):
    total_time = time.time()
    # 사투리 결정
    if model_type == 0:  # 제주도
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type == 1:  # 경상도
        korean2dialect = gyeong_translation
        text2speech = gyeong_speech
    elif model_type == 2:  # 전라도
        korean2dialect = jeon_translation
        text2speech = jeon_speech
    else:
        print(model_type)
        raise NotImplementedError

    dialect = korean2dialect.transform(korean)
    # dialect = korean
    translated_length = min(200,int(len(korean) * 2))
    dialect = dialect[:translated_length] if len(dialect) > translated_length else dialect  # 번역
    print(f'translated text: {dialect}')
    txt_list = clean_text(txt=dialect)  # 번역된 텍스트 클리닝
    txt_list = txt_list[:4] if len(''.join(txt_list)) > translated_length else txt_list  # 번역
    try:
        wav_file, error_log = text2speech.forward(txt_list)  # 텍스트 -> wav file
        error_sentences = []
        for k, v in error_log.items():
            if v is True:
                error_sentences.append(k)
        error_sentences = '|'.join(error_sentences)
        return_data = {'translated_text': dialect, 'audio_stream': wav_file}
        res = app.response_class(response=json.dumps(return_data), status=200, mimetype='application/json')
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        print('Total time(translation + synthesize): {}'.format(time.time() - total_time))
        try:
            from source.db.config import DB_URL
            from flask_sqlalchemy import SQLAlchemy
            from source.db.app import TTS
            app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
            tts = TTS(dialect_type=model_type, korean=korean, dialect=dialect, ip=ip, error=error_sentences)
            db = SQLAlchemy(app)
            db.session.add(tts)
            db.session.commit()
        except Exception as sql_e:
            print(sql_e)
            pass
        dialect, txt_list, wav_file, error_sentences, return_data, korean2dialect, text2speech = None, None, None, None, None, None, None
        torch.cuda.empty_cache()

    except Exception as e:
        print(e)
        dialect, txt_list, wav_file, error_sentences, return_data, korean2dialect, text2speech = None, None, None, None, None, None, None
        res = app.response_class(response=None, status=500, mimetype='application/json')
        torch.cuda.empty_cache()
    return res


@app.route('/ml-inference', methods=['POST'])
def ml_inference():
    print('====== Synthesizing ======')
    print(request.form)
    gender = int(request.form['gender'])  # [0, 1] == ['남자', '여자']
    model_type = int(request.form['model'])  # [0, 1, 2] = ['제주도', '경상도', '전라도]
    korean = request.form['input-text']  # 표준어 Input
    res = tts_inference(gender, model_type, korean)
    return res


@app.route('/api-inference', methods=['POST'])
def api_inference():
    print('====== Synthesizing ======')
    data = request.get_json()
    print(data)
    gender = int(data['gender'])  # [0, 1] == ['남자', '여자']
    model_type = int(data['model'])  # [0, 1, 2] = ['제주도', '경상도', '전라도]
    korean = data['input-text']  # 표준어 Input
    res = tts_inference(gender, model_type, korean)
    return res


@app.route('/')
def index():
    return render_template('index.html')
