import time
import random
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
random.seed(0)
np.random.seed(0)


def main():
    target_address = os.path.join(Path(os.getcwd()).parent,
                                  'Window_capture\\Data\\command_keys.npy')
    screenshot_address = os.path.join(Path(os.getcwd()).parent,
                                      'Window_capture\\Data\\screenshots.npy')
    labels = np.load(target_address)
    images = np.load(screenshot_address, allow_pickle=True)
    # labels = labels[14000:]
    # images = images[14000:, :]
    # Let's get rid of some -1 values.
    res_list = [i for i, value in enumerate(labels) if value == -1]
    # Randomly choose X number of entries to be deleted specified as -1
    idx = np.random.choice(res_list, 10000, replace=False)  # took 5 hours
    # flatten images then converted to dataframe for easier removal of idx
    images = pd.DataFrame(images)
    # flatten images then converted to dataframe for easier removal of idx
    images = np.array(images.drop(images.index[idx]))
    labels = np.delete(labels, idx)
    print(images.shape, labels.shape)
    print(np.unique(labels, return_counts=True))
    print("Length Command Keys Shape: ", labels.shape)
    print("Length Screenshot Shape: ", images.shape)
    print("Screenshot Shape: ", images[0].shape)
    print(np.unique(labels, return_counts=True))

    print("Beginning SMOTE")
    smote = SMOTE(random_state=101)
    images, labels = smote.fit_resample(images, labels)
    print("Post SMOTE:", np.unique(labels, return_counts=True))
    # Cast -1 to 0, 38 to 1 and 40 to 2
    labels[labels == -1] = 0
    labels[labels == 38] = 1
    labels[labels == 40] = 2
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25)
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)
    start_time = time.time()
    xgb_model = XGBClassifier(use_label_encoder=False,
                              eval_metric='mlogloss')

    parameters = {'objective': ['multi:softmax'],
                  'learning_rate': [0.05, 0.1],  # so called `eta` value
                  'max_depth': [7, 9],
                  'min_child_weight': [3, 5],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [100],  # 1000? for better results?
                  }

    clf = GridSearchCV(xgb_model,
                       parameters,
                       n_jobs=5,
                       cv=StratifiedKFold(n_splits=5),
                       scoring='neg_log_loss',
                       verbose=2,
                       refit=True)

    clf.fit(X_train, y_train)
    print("XGBoost (no wrapper) Time: {}s".format(time.time() - start_time))
    print("Best model: ", clf.best_estimator_)
    preds = np.round(clf.best_estimator_.predict(X_test))
    acc = 1. - (np.abs(preds - y_test).sum() / y_test.shape[0])
    print("Acc: {}".format(acc))
    y_hat = clf.best_estimator_.predict(X_test)
    print(f'LogReg accuracy on held-out frames = {round(accuracy_score(y_test, y_hat),4)}')
    confusion_matrix(y_test, y_hat, labels=[0, 1, 2])
    target_names = ['0', '1', '2']
    print(classification_report(y_test, y_hat, target_names=target_names))
    pickle.dump(clf.best_estimator_, open('Existing_Models/xgboost_dino_tuned.pkl', 'wb'))


if __name__ == '__main__':
    main()
