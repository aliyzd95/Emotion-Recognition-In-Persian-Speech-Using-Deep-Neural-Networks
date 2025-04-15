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
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
import json
import warnings
warnings.filterwarnings("ignore")

# لیست نام احساسات (بدون نمونه "fear")
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral"]

# بارگذاری تقسیم‌بندی‌های از پیش تهیه شده (folds) از فایل pickle
with open("files/modified_folds.pickle", "rb") as of:
    modified_folds = pickle.load(of)

def generate_folds(X, y, n=5):
    """
    تقسیم داده‌ها به n fold با استفاده از StratifiedKFold جهت حفظ توزیع کلاس‌ها.
    خروجی تقسیم‌بندی‌ها به صورت فایل‌های JSON و pickle ذخیره می‌شوند.
    """
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed_value)
    folds = {}
    count = 1
    for train, test in kfold.split(X, y):
        folds[f'fold_{count}'] = {'train': train.tolist(), 'test': test.tolist()}
        print(len(folds[f'fold_{count}']['train']))
        count += 1
    print(len(folds) == n)
    with open('files/modified_folds.json', 'w') as fj:
        json.dump(folds, fj)
    with open('files/modified_folds.pickle', 'wb') as fp:
        pickle.dump(kfold, fp)
        print('_____________________ saved! _____________________')
    with open('files/modified_folds.json') as f:
        kfolds = json.load(f)
    for key, val in kfolds.items():
        print(key, val)

def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    """
    تولید و نمایش نمودار ماتریس اشتباهات.
    در صورت انتخاب normalize، ماتریس به صورت نسبی نمایش داده می‌شود.
    """
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

def plot_confusion_matrix(predicted_labels_list, y_test_list):
    """
    محاسبه و رسم ماتریس اشتباهات با استفاده از لیست برچسب‌های پیش‌بینی شده و واقعی.
    """
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=emo_labels, normalize=True, title='SVM + eGeMAPS')
    plt.show()

def svm(X, y):
    """
    آموزش مدل SVM برای طبقه‌بندی احساسات با استفاده از یک رویکرد تو در تو:
      - cv_outer: تقسیم‌بندی‌های خارجی (از قبل تهیه شده)
      - cv_inner: تقسیم‌بندی‌های داخلی جهت جستجوی بهینه پارامترها (10 fold)
    از OneVsOneClassifier برای مقابله با مشکل چندکلاسه بودن استفاده می‌شود.
    جستجوی بهینه بر روی پارامترهای C و gamma با استفاده از BayesSearchCV انجام می‌شود.
    یک pipeline شامل StandardScaler برای نرمال‌سازی ویژگی‌ها به کار رفته و در نهایت نتایج cross-validation گزارش می‌شوند.
    """
    cv_outer = modified_folds
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    model = SVC()
    ovo = OneVsOneClassifier(model)
    space = {
        'estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5],
        'estimator__gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    }
    search = BayesSearchCV(ovo, space, scoring='recall_macro', cv=cv_inner, n_jobs=-1, verbose=0)
    pipeline = make_pipeline(StandardScaler(), search)
    scores = cross_validate(pipeline, X, y,
                            scoring=['recall_macro', 'accuracy'],
                            cv=cv_outer, n_jobs=-1, verbose=2)
    print('____________________ Support Vector Machine ____________________')
    print(f"Weighted Accuracy: {np.mean(scores['test_accuracy'] * 100)}")
    print(f"Unweighted Accuracy: {np.mean(scores['test_recall_macro']) * 100}")

# بارگذاری ویژگی‌های استخراج شده از فایل (به عنوان مثال از opensmile) و برچسب‌های مربوطه
X = np.load('files/modified_opensmile_eGeMAPS_features.npy').squeeze()
N_FEATURES = X.shape[1]
y = np.load('files/modified_emotions.npy')

# مخلوط‌سازی تصادفی نمونه‌ها برای از بین بردن ترتیب احتمالی در داده‌ها
N_SAMPLES = X.shape[0]
perm = np.random.permutation(N_SAMPLES)
X = X[perm]
y = y[perm]

if __name__ == '__main__':
    generate_folds(X, y)
    svm(X, y)

    # modified
    # Accuracy: 76.14503665093885
    # UAR: 61.240183756698016

    # old
    # Accuracy: 72.95645139237044
    # UAR: 58.66650383394545
