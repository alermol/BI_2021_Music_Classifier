#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from prettytable import PrettyTable

import config

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description=(
        'Script for genres Music Classification (MusiCl)\n\n'
        'Reconizible genres: Country music, Pop music, Hip hop music, '
        'Rock music, Metal, Classical music, Electro'
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        'Authors: Aleksey Ermolaev, Katerina Danko, Daria Andreeva, '
        'Evgenia Khokhlova, Aleksandr Andreev, Daniil Panshin'
    )
)
parser.add_argument('-v', '--version',
                    help='show version',
                    action='version',
                    version=f'Current version is {config.VERSION}')
parser.add_argument('path',
                    help='path to song',
                    metavar='path',
                    type=str)


def count_features(file):
    y, sr = librosa.load(file, sr=None)
    S = np.abs(librosa.stft(y))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    result = pd.DataFrame({
        'tempo': [tempo],
        'total_beats': [np.sum(beats)],
        'average_beats': [np.average(beats)],
        'chroma_stft_mean': [np.mean(chroma_stft)],
        'chroma_stft_std': [np.std(chroma_stft)],
        'chroma_stft_var': [np.var(chroma_stft)],
        'chroma_cq_mean': [np.mean(chroma_cq)],
        'chroma_cq_std': [np.std(chroma_cq)],
        'chroma_cq_var': [np.var(chroma_cq)],
        'chroma_cens_mean': [np.mean(chroma_cens)],
        'chroma_cens_std': [np.std(chroma_cens)],
        'chroma_cens_var': [np.var(chroma_cens)],
        'melspectrogram_mean': [np.mean(melspectrogram)],
        'melspectrogram_std': [np.std(melspectrogram)],
        'melspectrogram_var': [np.var(melspectrogram)],
        'mfcc_mean': [np.mean(mfcc)],
        'mfcc_std': [np.std(mfcc)],
        'mfcc_var': [np.var(mfcc)],
        'mfcc_delta_mean': [np.mean(mfcc_delta)],
        'mfcc_delta_std': [np.std(mfcc_delta)],
        'mfcc_delta_var': [np.var(mfcc_delta)],
        'rmse_mean': [np.mean(rmse)],
        'rmse_std': [np.std(rmse)],
        'rmse_var': [np.var(rmse)],
        'cent_mean': [np.mean(cent)],
        'cent_std': [np.std(cent)],
        'cent_var': [np.var(cent)],
        'spec_bw_mean': [np.mean(spec_bw)],
        'spec_bw_std': [np.std(spec_bw)],
        'spec_bw_var': [np.var(spec_bw)],
        'contrast_mean': [np.mean(contrast)],
        'contrast_std': [np.std(contrast)],
        'contrast_var': [np.var(contrast)],
        'rolloff_mean': [np.mean(rolloff)],
        'rolloff_std': [np.std(rolloff)],
        'rolloff_var': [np.var(rolloff)],
        'poly_mean': [np.mean(poly_features)],
        'poly_std': [np.std(poly_features)],
        'poly_var': [np.var(poly_features)],
        'tonnetz_mean': [np.mean(tonnetz)],
        'tonnetz_std': [np.std(tonnetz)],
        'tonnetz_var': [np.var(tonnetz)],
        'zcr_mean': [np.mean(zcr)],
        'zcr_std': [np.std(zcr)],
        'zcr_var': [np.var(zcr)],
        'harm_mean': [np.mean(harmonic)],
        'harm_std': [np.std(harmonic)],
        'harm_var': [np.var(harmonic)],
        'perc_mean': [np.mean(percussive)],
        'perc_std': [np.std(percussive)],
        'perc_var': [np.var(percussive)],
        'frame_mean': [np.mean(frames_to_time)],
        'frame_std': [np.std(frames_to_time)],
        'frame_var': [np.var(frames_to_time)]
    })
    return result


def print_result_table(table):
    result_table = PrettyTable()
    result_table.field_names = ('Genre', 'Probability')
    for row in table.itertuples(index=False):
        result_table.add_row((row[0], f'{row[1]}%'))
    return result_table


def main():
    args = parser.parse_args()
    clf = load('RF_classifier.joblib')
    print('Feature counting...', end='\r')
    features_vector = count_features(args.path)
    proba = pd.DataFrame({'cls': clf.classes_,
                          'prob': clf.predict_proba(features_vector)[0]})
    proba['cls'].replace({
        'country_group': 'Country music',
        'pop': 'Pop music',
        'hip_hop': 'Hip hop music',
        'rock': 'Rock music',
        'metal': 'Metal',
        'classic': 'Classical music',
        'electro': 'Electro'
    }, inplace=True)
    proba.sort_values('prob', ascending=False, inplace=True)
    proba['prob'] = proba['prob'].apply(lambda x: round(float(x) * 100, 2))
    result = (
        f'Probability of classes for song {Path(args.path).stem}\n'
        f'{print_result_table(proba)}'
    )
    print(result)


if __name__ == '__main__':
    main()
