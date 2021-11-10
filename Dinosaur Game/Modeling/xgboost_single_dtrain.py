import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
import time
import psutil


def main():
    target_address = 'command_keys.npy'
    screenshot_address = 'screenshots.npy'
    labels = np.load(target_address)
    images = np.load(screenshot_address, allow_pickle=True)

    print("Length Command Keys Shape: ", labels.shape)
    print("Length Screenshot Shape: ", images.shape)
    print("Screenshot Shape: ", images[0].shape)
    print(np.unique(labels, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25)
    print(np.unique(y_train, return_counts=True))
    print("Total Memory ",
          psutil.virtual_memory().total / (1024.0 ** 3), '\n',
          "Available Memory ",
          psutil.virtual_memory().available / (1024.0 ** 3), '\n',
          "Used Memory ",
          psutil.virtual_memory().used / (1024.0 ** 3), '\n',
          )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print('-----')
    print("Total Memory ",
          psutil.virtual_memory().total / (1024.0 ** 3), '\n',
          "Available Memory ",
          psutil.virtual_memory().available / (1024.0 ** 3), '\n',
          "Used Memory ",
          psutil.virtual_memory().used / (1024.0 ** 3), '\n',
          )
    start_time = time.time()
    param = {'min_child_weight': 10,
             'subsample': 0.7,
             'max_depth': 7,
             'learning_rate': 0.03,
             'colsample_bytree': 1,
             # 'n_gpus': 1,
             'alpha': 0,
             'lambda': 1,
             'max_bin': 256,
             # 'tree_method': 'gpu_hist',
             'objective': 'multi:softmax',
             'num_class': 3, }

    bst = xgb.train(param,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtest, "val")],
                    early_stopping_rounds=50)

    print("XGBoost (no wrapper) Time: {}s".format(time.time() - start_time))
    preds = np.round(bst.predict(dtest))
    acc = 1. - (np.abs(preds - y_test).sum() / y_test.shape[0])
    print("Acc: {}".format(acc))
    print("Prediction time --- %s seconds ---" % (time.time() - start_time))
    y_hat = bst.predict(dtest)
    print(f'LogReg accuracy on held-out frames = {round(accuracy_score(y_test, y_hat),4)}')
    confusion_matrix(y_test, y_hat, labels=[0, 1, 2])
    target_names = ['nothing', 'up', 'down']
    print(classification_report(y_test, y_hat, target_names=target_names))
    pickle.dump(bst, open('xgboost_dino_CLOUD_single.pkl', 'wb'))


if __name__ == '__main__':
    main()
