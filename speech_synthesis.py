# -*- coding: cp949 -*-
import base64
import sys
import time

from numpy import finfo

from source.distributed import apply_gradient_allreduce
from source.text import hangul_to_sequence
from source.utils import load_wav_to_torch
sys.path.append('source/waveglow/')
import torch
import numpy as np
from source.hparams import create_hparams
from source.waveglow.denoiser import Denoiser
from scipy.io import wavfile
from source.layers import TacotronSTFT
from source.model import Tacotron2

hparams = create_hparams()

def save_wav(wav, path, sr):
    wav = wav * 32767  / max(0.01, np.max(np.abs(wav)))
    wav = np.clip( wav * 1.2, -32767, 32767) ### double wave range to increase volume, clip to maximize to avoid lossing audio quality
    # proposed by @dsmiller
    wav = wav.astype(np.int16)
    wavfile.write(path, sr, wav)
    return wav

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model

def save_audio_to_mel(filename):
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,hparams.mel_fmax)
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    np_melspec = melspec.data.cpu().numpy()
    np.save(f'{filename.replace(".wav", ".npy")}', np_melspec, allow_pickle=True)
    print(f'{filename.replace(".wav", ".npy")}')


def load_preset_mel(folder, id=0):
    import os
    import numpy as np
    preset_mels = []
    for mel in os.listdir(folder):
        if mel.endswith('.npy'):
            mel_path = os.path.join(folder,mel)
            np_mel = np.load(mel_path, allow_pickle=False)
            preset_mels.append(np_mel)
    return preset_mels[id]

class Text2Speech(object):
    def __init__(self, model_type, gender):
        self.hparams = create_hparams()
        self.hparams.fp16_run=False
        self.hparams.distributed_run=False
        self.checkpoint_path, self.waveglow_path = self.select_model(model_type, gender)
        self.model = load_model(self.hparams)
        self.model.load_state_dict(torch.load(self.checkpoint_path)['state_dict'])
        self.waveglow = torch.load(self.waveglow_path)['model']

        self.model.cuda().eval().half()
        self.waveglow.cuda().eval().half()
        for k in self.waveglow.convinv:
            k.float()
        self.denoiser = Denoiser(self.waveglow)

    def forward(self, txt_list):
        silent = self.create_silent()
        mels, error = self.tacotron_synthesize(txt_list, silent)
        audio = self.wavenet_synthesize(mels)
        audio = self.denoiser(audio, strength=0.06)[:, 0]
        audio = audio[0].data.cpu().numpy()

        save_wav(path='output.wav', wav=audio, sr=hparams.sampling_rate)
        with open('output.wav', "rb") as binary_file:
            # Read the whole file at once
            output_wav = binary_file.read()

        # encode with base64 and response to client
        wav_file = base64.b64encode(output_wav).decode('utf-8')
        return wav_file, error

    def create_silent(self):
        silent = torch.zeros(torch.Size([40, 80, 1]),
                             dtype=torch.half).cuda() if self.hparams.fp16_run else torch.zeros(
            torch.Size([40, 80, 1]), dtype=torch.float).cuda()
        silent = torch.add(silent, -12)
        return silent

    def tacotron_synthesize(self, txt_list, silent):
        mels = None
        start = time.time()
        error = {}
        # preset_mels = load_preset_mel('source/filelists/preset')

        for i, text in enumerate(txt_list):
            torch.cuda.empty_cache()
            sequence = np.array(hangul_to_sequence(text))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
            ### calculate audio length
            audio_length = mel_outputs_postnet.shape[-1] * hparams.hop_length / hparams.sampling_rate  #### by second
            max_length = 46
            # print(f'sequence_length {sequence_length}')
            # print(f'audio length {audio_length}')
            # print(f'pred_audio_length - np.sqrt(0.208) {pred_audio_length - 2}')
            # print(f'pred_audio_length + np.sqrt(0.208) {pred_audio_length + 2}')
            if (audio_length >= max_length):
                error[text] = True
                from random import randint
                if hparams.fp16_run:
                    mel_outputs_postnet = torch.from_numpy(load_preset_mel(folder=r'source/preset_audio', id=randint(0,10))).unsqueeze(0).cuda().half()
                else:
                    mel_outputs_postnet = torch.from_numpy(load_preset_mel(folder=r'source/preset_audio', id=randint(0, 10))).unsqueeze(0).cuda().float()
            else:
                error[text] = False

            if i < len(txt_list) - 1:
                mels = torch.cat(
                    [mels, mel_outputs_postnet.transpose(2, 0), silent]) if mels is not None else torch.cat(
                    [mel_outputs_postnet.transpose(2, 0), silent])
            else:
                mels = torch.cat([mels, mel_outputs_postnet.transpose(2, 0)]) if mels is not None else torch.cat(
                    [mel_outputs_postnet.transpose(2, 0)])

        print('Tacotron synthesize time: {}'.format(time.time() - start))
        del sequence, mel_outputs, mel_outputs_postnet, audio_length, max_length, alignments
        return mels, error

    def wavenet_synthesize(self, mels):
        start = time.time()
        with torch.no_grad():
            audio = self.waveglow.infer(mels.transpose(0, 2), sigma=0.6666)
        print('Wavenet synthesize time: {}'.format(time.time() - start))
        return audio

    def select_model(self, model_type,gender):
        if gender == 'male':  ### male
            if model_type == '제주':
                return self.hparams.t_path_jeju_male, self.hparams.w_jeju_path_male
            elif model_type == '경상':
                return self.hparams.t_path_gyeongsang_male, self.hparams.w_gyeongsang_path_male
            elif model_type == '전라':
                return self.hparams.t_path_jeon_male, self.hparams.w_jeon_path_male
            elif model_type == '연변':
                return self.hparams.t_path_yeonbyeon_male, self.hparams.w_yeonbyeon_path_male
            elif model_type == '표준':
                return self.hparams.t_path_standard_male, self.hparams.w_jeju_standard_male
            else:
                raise NotImplementedError
        else:  ### female
            if model_type == '제주':
                return self.hparams.t_path_jeju_female, self.hparams.w_jeju_path_female
            elif model_type == '경상':
                return self.hparams.t_path_gyeongsang_female, self.hparams.w_gyeongsang_path_female
            elif model_type == '전라':
                return self.hparams.t_path_jeon_female, self.hparams.waveglow_jeonla_female
            elif model_type == '연변':
                return self.hparams.t_path_yeonbyeon_female, self.hparams.w_yeonbyeon_path_female
            elif model_type == '표준':
                return self.hparams.t_path_standard_female, self.hparams.w_jeju_standard_female
            else:
                raise NotImplementedError
