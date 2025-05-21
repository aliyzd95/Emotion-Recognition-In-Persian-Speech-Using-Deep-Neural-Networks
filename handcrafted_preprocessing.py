seed_value = 42

import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import pickle
import math
import librosa
from keras_preprocessing.sequence import pad_sequences

sr = 16000
duration = 7.52
n_fft = 512 
hop_length = 256 
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]
path = "ShEMO" 


def get_emotion_label(file_name):
    return emo_codes[file_name[3]]


def get_emotion_name(file_name):
    return emo_labels[emo_codes[file_name[5]]]


def read_files():
    wavs = []
    print('---------- reading files ----------')
    for file in os.listdir(path):
        y, _ = librosa.load(f'{path}/{file}', sr=sr, mono=True, duration=duration)
        wavs.append(y)
    wavs_padded = pad_sequences(wavs, maxlen=int(sr * duration), dtype="float32")
    with open('files/wavs_padded.pickle', 'wb') as wp:
        pickle.dump(wavs_padded, wp)


def feature_extraction():
    with open('files/wavs_padded.pickle', 'rb') as wp:
        wavs_padded = pickle.load(wp)
        print(f'waves padded shape: {wavs_padded.shape}')
    features = []
    print('---------- extracting features ----------')
    N_FRAMES = math.ceil(wavs_padded.shape[1] / hop_length)
    print(f'n_frames: {N_FRAMES}')
    for y, name in zip(wavs_padded, os.listdir(path)):
        frames = []
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        poly_features = librosa.feature.poly_features(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        S, phase = librosa.magphase(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, S=S)[0]
        mfcc = librosa.feature.mfcc(y=y, n_fft=n_fft, sr=sr, hop_length=hop_length)
        mfcc_der = librosa.feature.delta(mfcc)
        for i in range(N_FRAMES + 1):
            f = [zero_crossing_rate[i], rms[i], spectral_flatness[i],
                 spectral_centroid[i], spectral_bandwidth[i], spectral_rolloff[i]]
            for sc in spectral_contrast[:, i]:
                f.append(sc)
            for m_coeff in mfcc[:, i]:
                f.append(m_coeff)
            for m_coeff_der in mfcc_der[:, i]:
                f.append(m_coeff_der)
            for ch_st in chroma_stft[:, i]:
                f.append(ch_st)
            for ch_ce in chroma_cens[:, i]:
                f.append(ch_ce)
            for ch_cq in chroma_cqt[:, i]:
                f.append(ch_cq)
            for p in poly_features[:, i]:
                f.append(p)
            frames.append(f)
        features.append(frames)
    features = np.array(features)
    np.save('files/handcrafted_features-7.52s-32ms.npy', features)
    print(f'features shape: {features.shape}')


if __name__ == '__main__':
    read_files()
    feature_extraction()
