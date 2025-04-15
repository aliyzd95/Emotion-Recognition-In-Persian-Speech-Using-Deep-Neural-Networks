seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import opensmile  # کتابخانه استخراج ویژگی‌های صوتی
import pickle
import json

# تنظیمات اولیه
path = "ShEMO"
duration = 7.52
sr = 16000
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4, "fear": 5}
emo_names = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]

# بارگذاری اطلاعات تغییر یافته دیتاست از فایل JSON
with open('files/modified_shemo.json', encoding='utf-8') as fd:
    modified_shemo = json.loads(fd.read())


def opensmile_Functionals(dataset):
    # استخراج ویژگی‌های Functionals (eGeMAPSv02) از فایل‌های صوتی
    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
    )
    features = []
    for data in dataset:
        d = dataset[data]
        if d["emotion"] != 'fear':  # حذف نمونه‌های "fear"
            df = feature_extractor.process_file(d["path"])
            features.append(df)
    features = np.array(features).squeeze()
    print(features.shape)
    np.save('files/modified_opensmile_eGeMAPS_features.npy', features)


def opensmile_LLDs():
    # استخراج ویژگی‌های سطح پایین (LLDs) و دلتای آن‌ها از ComParE_2016
    feature_extractor_1 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        verbose=True, num_workers=None
    )
    feature_extractor_2 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
        verbose=True, num_workers=None
    )
    with open('files/wavs_padded.pickle', 'rb') as wp:
        wavs_padded = pickle.load(wp)
    features = []
    for wav in wavs_padded:
        df_1 = feature_extractor_1.process_signal(wav, sr)
        df_2 = feature_extractor_2.process_signal(wav, sr)
        feature_1 = np.array(df_1)
        feature_2 = np.array(df_2)
        # ادغام ویژگی‌های اصلی و دلتای آن‌ها (با حذف دو سطر آخر از دلتای ویژگی‌ها)
        feature = np.concatenate((feature_1, feature_2[:-2, ]), axis=1)
        features.append(feature)
    features = np.array(features)
    np.save('files/opensmile_eGeMAPS_lld_features.npy', features)


def opensmile_cmd(dataset):
    # اجرای ابزار خط فرمان SMILExtract برای استخراج ویژگی‌های emo_large
    for data in dataset:
        d = dataset[data]
        if d['emotion'] != "fear":
            opensmile_config_path = 'opensmile-3.0.1-win-x64/config/misc/emo_large.conf'
            single_feat_path = 'files/modified_speech_features.csv'
            cmd = f'SMILExtract -C {opensmile_config_path} -I {d["path"]} -O {single_feat_path}'
            os.system(cmd)

    # پردازش فایل خروجی جهت استخراج ویژگی‌ها
    this_path_output = 'files/modified_speech_features.csv'
    with open(this_path_output, encoding='utf-8') as f:
        lines = []
        for line in f.readlines()[6559:]:  # رد کردن بخش header
            lines.append(line.split(',')[1:-1])
    features = np.array(lines, dtype='float32')
    print(features.shape)
    np.save('files/modified_opensmile_emolarge_features.npy', features)


def emotions(dataset):
    # استخراج برچسب‌های عددی احساسات (به جز "fear")
    emotions = []
    for data in dataset:
        d = dataset[data]
        if d["emotion"] != 'fear':
            emotions.append(emo_labels[d["emotion"]])
    emotions = np.array(emotions)
    print(emotions.shape)
    np.save('files/modified_emotions.npy', emotions)


if __name__ == '__main__':
    # اجرای استخراج ویژگی‌ها از طریق خط فرمان و ذخیره برچسب‌ها
    # opensmile_Functionals(dataset=modified_shemo)
    opensmile_cmd(dataset=modified_shemo)
    emotions(dataset=modified_shemo)
