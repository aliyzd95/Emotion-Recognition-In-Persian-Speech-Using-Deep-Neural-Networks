seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import itertools
import pickle
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Dense, Input, MaxPooling1D, SpatialDropout1D, \
    GlobalAveragePooling1D
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
import seaborn
import matplotlib.pyplot as plt
from keras import backend as K


def non_nan_average(x):
    nan_mask = tf.math.is_nan(x)
    x = tf.boolean_mask(x, tf.logical_not(nan_mask))
    return K.mean(x)


def uar_accuracy(y_true, y_pred):
    pred_class_label = K.argmax(y_pred, axis=-1)
    true_class_label = y_true
    cf_mat = tf.math.confusion_matrix(true_class_label, pred_class_label)
    diag = tf.linalg.tensor_diag_part(cf_mat)
    total_per_class = tf.reduce_sum(cf_mat, axis=1)
    acc_per_class = diag / tf.maximum(1, total_per_class)
    return non_nan_average(acc_per_class)


with open("files/modified_folds.pickle", "rb") as of:
    modified_folds = pickle.load(of)

MODEL = 'acoustic'
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral"]


def test_model(units=512):
    input_speech = Input((1, N_FEATURES))
    speech = Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(input_speech)
    speech = MaxPooling1D(padding='same')(speech)
    speech = BatchNormalization()(speech)
    speech = SpatialDropout1D(0.5)(speech)
    speech = Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(speech)
    speech = MaxPooling1D(padding='same')(speech)
    speech = BatchNormalization()(speech)
    speech = SpatialDropout1D(0.5)(speech)
    speech = GlobalAveragePooling1D()(speech)

    output_speech = Dense(5, activation='softmax')(speech)
    model = Model(input_speech, output_speech)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', uar_accuracy])
    plot_model(model, to_file='files/model.png', show_shapes=True)
    return model


def fit_model(X, y):
    uar_per_fold = []
    acc_per_fold = []
    loss_per_fold = []
    kfold = modified_folds
    fold_no = 1
    for train, test in kfold.split(X, y):
        X_train, X_test = X[train], X[test]

        # نرمال‌سازی داده‌ها
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

        model = test_model()

        best_weights_file = f"files/{MODEL}_weights_{fold_no}.h5"
        es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
        mc = ModelCheckpoint(best_weights_file, monitor='val_accuracy', verbose=1, save_best_only=True)

        history = model.fit(
            X_train, y[train],
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[es, mc],
            verbose=1
        )
        scores = model.evaluate(X_test, y[test], verbose=0)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        uar = recall_score(y[test], y_pred, average='macro')
        uar_per_fold.append(scores[2])
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no += 1

    print('____________________ RESULTS ____________________')
    print(f'> Accuracy: {np.mean(acc_per_fold) * 100} (+- {np.std(acc_per_fold) * 100})')
    print(f'> UAR: {np.mean(uar_per_fold) * 100} (+- {np.std(uar_per_fold) * 100})')
    print(f'> Loss: {np.mean(loss_per_fold)}')


X = np.load('files/modified_opensmile_emolarge_features.npy')
N_FEATURES = X.shape[1]
y = np.load('files/modified_emotions.npy')

N_SAMPLES = X.shape[0]
perm = np.random.permutation(N_SAMPLES)
X = X[perm]
y = y[perm]

if __name__ == '__main__':
    fit_model(X, y)

# Average scores for all folds:
# > accuracy: 79.68271851539612 (+- 1.5159805175458114)
# > UAR: 66.12497121309126 (+- 3.786788830359496)
# > loss: 0.9580800771713257
