seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import keras
from keras import backend as K
import itertools
import pickle
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM, Input, Masking, \
    MaxPooling1D, Concatenate, Lambda, Dot, Softmax
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from keras.optimizers import Adam
import seaborn
import matplotlib.pyplot as plt

with open("files/outer_folds.pickle", "rb") as of:
    outer_folds = pickle.load(of)

MODEL = 'TEST'
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral"]


def test_model(units=512, drop_rate=0.4):
    inp = Input(shape=(N_FRAMES, N_FEATURES))
    x = Masking()(inp)
    x = Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    x = Conv1D(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    states, forward_h, _, backward_h, _ = Bidirectional(LSTM(units, return_sequences=True, return_state=True))(x)
    last_state = Concatenate()([forward_h, backward_h])
    hidden = Dense(units, activation='tanh', use_bias=False,
                   kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(states)
    out = Dense(1, activation='linear', use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(hidden)
    flat = Flatten()(out)
    energy = Lambda(lambda t: t / np.sqrt(units))(flat)
    alpha = Softmax(name="alpha")(energy)
    context_vector = Dot(axes=1)([states, alpha])
    context_vector = Concatenate()([context_vector, last_state])
    pred = Dense(5, activation="softmax")(context_vector)
    model = keras.Model(inputs=inp, outputs=pred)

    # DNN
    # model = Sequential()
    # model.add(Dense(256, activation='relu', input_shape=(N_FRAMES, N_FEATURES)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(5, activation='softmax'))

    # BLSTM
    # model = Sequential()
    # model.add(layers.Bidirectional(layers.LSTM(units, input_shape=(N_FRAMES, N_FEATURES), return_sequences=False)))
    # model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='results/model.png', show_shapes=True)
    print(model.summary())
    return model


def report(real_class, pred_class):
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:\n" + str(cm) + "\n")
    data = np.array(cm).flatten().reshape(5, 5)
    plt.title('CNN-BLSTM')
    seaborn.heatmap(cm, xticklabels=emo_labels, yticklabels=emo_labels, annot=data, cmap="Blues")
    plt.savefig(MODEL + "_conf_matrix.png")
    plt.gcf().clear()


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cnf_matrix


def plot_confusion_matrix(predicted_labels_list, y_test_list, title):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=emo_labels, normalize=True, title=title)
    plt.show()


def fit_model(X, y):
    uar_per_fold = []
    acc_per_fold = []
    loss_per_fold = []
    predicted_targets = np.array([])
    actual_targets = np.array([])
    kfold = outer_folds
    fold_no = 1
    for train, test in kfold.split(X, y):
        X_train, X_test = X[train], X[test]
        scaler = StandardScaler()
        X_train = X_train.reshape((X_train.shape[0], N_FRAMES * N_FEATURES))
        X_train = scaler.fit_transform(X_train)
        X_train = X_train.reshape((X_train.shape[0], N_FRAMES, N_FEATURES))
        X_test = X_test.reshape((X_test.shape[0], N_FRAMES * N_FEATURES))
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape((X_test.shape[0], N_FRAMES, N_FEATURES))
        model = test_model()
        best_weights_file = "files/openSMILE_" + MODEL + "_weights.h5"
        es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
        mc = ModelCheckpoint(best_weights_file, monitor='val_accuracy', verbose=1, save_best_only=True)
        history = model.fit(X_train, y[train],
                            validation_data=(X_test, y[test]),
                            epochs=100,
                            batch_size=32,
                            callbacks=[es, mc],
                            verbose=1)
        scores = model.evaluate(X_test, y[test], verbose=0)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        predicted_targets = np.append(predicted_targets, y_pred)
        actual_targets = np.append(actual_targets, y[test])
        uar = recall_score(y[test], y_pred, average='macro')
        uar_per_fold.append(uar)
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        fold_no += 1
    for i in range(len(acc_per_fold)):
        print(f'> fold {i + 1} - uar: {uar_per_fold[i]} - accuracy: {acc_per_fold[i]} - loss: {loss_per_fold[i]}')
    print('____________________ RESULTS ____________________')
    print('Average scores for all folds:')
    print(f'> accuracy: {np.mean(acc_per_fold) * 100} (+- {np.std(acc_per_fold) * 100})')
    print(f'> UAR: {np.mean(uar_per_fold) * 100} (+- {np.std(uar_per_fold) * 100})')
    print(f'> loss: {np.mean(loss_per_fold)}')
    plot_confusion_matrix(predicted_targets, actual_targets, 'hand_crafted_LLDs - CNN + attention-BLSTM')


features = np.load('files/handcrafted_features-7.52s-32ms.npy')
print(features.shape)
N_FRAMES = features.shape[1]
N_FEATURES = features.shape[2]
emotions = np.load('files/emotions.npy')

N_SAMPLES = len(features)
perm = np.random.permutation(N_SAMPLES)
features = features[perm]
emotions = emotions[perm]

X = []
y = []
for f, e in zip(features, emotions):
    if e != 5:
        X.append(f)
        y.append(e)
X = np.asarray(X)
y = np.asarray(y)

if __name__ == '__main__':
    fit_model(X, y)
