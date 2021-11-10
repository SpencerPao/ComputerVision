"""This script is used to find best parameters for XGBoost using BayesianOptimization."""
# https://machinelearningapplied.com/hyperparameter-search-with-gpyopt-part-2-xgboost-classification-and-ensembling/
import os
import numpy as np
import pandas as pd
import GPyOpt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import joblib
import time
from pathlib import Path

# *******************************************************************


def create_save_models(dtrain,
                       dvalid,
                       error_metric,
                       num_random_points,
                       num_iterations,
                       results_dir,
                       processor_type,
                       number_of_classes):
    """Create and save models trained from Bayesian Optimization."""
    dict_single_parameters = {'booster': 'gbtree',
                              'verbosity': 0,
                              'objective': 'multi:softprob',
                              'eval_metric': error_metric,
                              'num_class': number_of_classes}

    if processor_type == 'cpu':
        tree_method = 'hist'
        predictor = 'auto'
    elif processor_type == 'gpu':
        tree_method = 'gpu_hist'
        predictor = 'gpu_predictor'
    else:
        print('\ninvalid processor_type in create_models():', processor_type)
        raise NameError

    dict_single_parameters['tree_method'] = tree_method
    dict_single_parameters['predictor'] = predictor

    # save to results dir
    dict_file = open(results_dir + 'dict_single_parameters.pkl', 'wb')
    pickle.dump(dict_single_parameters, dict_file)

    maximum_boosting_rounds = 1500
    early_stop_rounds = 10
    list_boosting_rounds = []
    list_model_name = []

    # create search space
    search_space = [
        {'name': 'eta', 'type': 'continuous', 'domain': (0.01, 0.3)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (100, 300, 500)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 2, 3)},
        {'name': 'max_depth', 'type': 'continuous', 'domain': (3, 21)},
        {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.5, 0.99)},
        {'name': 'reg_lambda', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 6)},
        {'name': 'reg_alpha', 'type': 'discrete', 'domain': (0, 1, 2, 3)},
        {'name': 'subsample', 'type': 'continuous', 'domain': (0.6, 0.9)},
        {'name': 'max_bin', 'type': 'discrete',
         'domain': (50, 75, 100, 125, 150, 200, 250)}]

    if error_metric == 'mlogloss':
        maximize_bool = False
    else:
        raise NameError

    start_time_total = time.time()

    def xgb_function(params):

        dict_parameters = {'eta': params[0][0],
                           'n_estimators': params[0][1],
                           'min_child_weight': params[0][2],
                           'max_depth': int(params[0][3]),
                           'colsample_bytree': params[0][4],
                           'reg_lambda': int(params[0][5]),
                           'reg_alpha': int(params[0][6]),
                           'subsample': params[0][7],
                           'max_bin': int(params[0][8])}

        dict_parameters.update(dict_single_parameters)
        print(dict_parameters)
        watchlist = [(dvalid, 'eval')]

        xgb_model = xgb.train(params=dict_parameters, dtrain=dtrain, evals=watchlist,
                              num_boost_round=maximum_boosting_rounds,
                              early_stopping_rounds=early_stop_rounds,
                              verbose_eval=False)

        model_name = 'xgb_gpyopt_' + str(np.random.uniform(0, 1,))[2:9]
        multiclass_log_loss = xgb_model.best_score
        boosting_rounds = xgb_model.best_ntree_limit

        print('\nmodel name:', model_name)
        print('multiclass_log_loss =', multiclass_log_loss)
        print('boosting_rounds = ', boosting_rounds)

        list_boosting_rounds.append(boosting_rounds)
        list_model_name.append(model_name)

        # save model
        if xgb_model.best_ntree_limit <= maximum_boosting_rounds:
            joblib.dump(xgb_model, results_dir + model_name + '.joblib')

        return multiclass_log_loss

    gpyopt_bo = GPyOpt.methods.BayesianOptimization(f=xgb_function,
                                                    domain=search_space,
                                                    model_type='GP',
                                                    initial_design_numdata=num_random_points,
                                                    initial_design_type='random',
                                                    acquisition_type='EI',
                                                    normalize_Y=True,
                                                    exact_feval=False,
                                                    acquisition_optimizer_type='lbfgs',
                                                    model_update_interval=1,
                                                    evaluator_type='sequential',
                                                    batch_size=1,
                                                    num_cores=os.cpu_count(),
                                                    verbosity=False,
                                                    verbosity_model=False,
                                                    maximize=maximize_bool,
                                                    de_duplication=True)

    # text file with parameter info, gp info, best results
    rf = results_dir + 'report'

    # text file with header: Iteration  Y   var_1   var_2   etc.
    ef = results_dir + 'evaluation'

    # text file with gp model info
    mf = results_dir + 'models'
    gpyopt_bo.run_optimization(max_iter=num_iterations,
                               report_file=rf,
                               evaluations_file=ef,
                               models_file=mf)

    elapsed_time_total = (time.time()-start_time_total)/60
    print('\n\ntotal elapsed time =', elapsed_time_total, ' minutes')

    # put results in dfs
    header_params = []
    for param in search_space:
        header_params.append(param['name'])

    df_results = pd.DataFrame(data=gpyopt_bo.X, columns=header_params)

    # integer values are rendered as real (1 -> 1.0)
    df_results[error_metric] = gpyopt_bo.Y
    df_results['boosting rounds'] = list_boosting_rounds
    df_results['model name'] = list_model_name
    df_results.to_pickle(results_dir + 'df_results.pkl')
    df_results.to_csv(results_dir + 'df_results.csv')

# end of create_save_models()

# *******************************************************************


def make_final_predictions(dcalib,
                           dprod,
                           yprod,
                           list_class_names,
                           models_directory,
                           save_directory,
                           save_models_flag, df_params,
                           threshold,
                           ml_name,
                           dict_single_params,
                           type_error,
                           accepted_models_num):
    """Get classification matrix from top N performing trained models."""
    if type_error == 'mlogloss':
        threshold_multiplier = 1.0
        df_params.sort_values(by=type_error, ascending=True, inplace=True)
    else:
        raise NameError

    # apply threshold
    accepted_models = 0
    list_predicted_prob = []
    num_models = df_params.shape[0]
    print(df_params)
    for i in range(num_models):
        bool1 = df_params.loc[df_params.index[i], type_error] < threshold_multiplier*threshold
        bool2 = accepted_models < accepted_models_num
        if bool1 and bool2:
            model_name = str(df_params.loc[df_params.index[i], 'model name'])
            try:
                xgb_model = joblib.load(models_directory + model_name + '.joblib')
                print("Loaded Model: ", model_name + '.joblib')
            except:
                print('\ncould not read', model_name)
            else:
                list_predicted_prob.append(xgb_model.predict(dprod).argmax(axis=1))
                # best result
                if i == 0:
                    target_names = ['nothing', 'up', 'down']
                    print(classification_report(
                        y_prod, list_predicted_prob[0], target_names=target_names))
                    name_result = ml_name + '_best'
                    joblib.dump(xgb_model, models_directory + name_result + '.joblib')
                    print("-------------------------------------------------------")
                    print("Saved Best Performing Model on Log Loss: ", name_result)
                    print("-------------------------------------------------------")
                else:
                    target_names = ['nothing', 'up', 'down']
                    print(classification_report(
                        y_prod, list_predicted_prob[0], target_names=target_names))

                accepted_models = accepted_models + 1


# *******************************************************************


if __name__ == '__main__':

    type_of_processor = 'cpu'  # Change to 'gpu' if you want to use gpu
    ml_algorithm_name = 'xgb'  # Change this for naming purposes.
    file_name_stub = 'gpyopt_' + ml_algorithm_name

    calculation_type = 'calibration'  # 'calibration' 'production'

    base_directory = os.getcwd()
    data_directory = os.path.join(Path(os.getcwd()).parent,
                                  'Window_capture\\Data\\')

    results_directory_stub = base_directory + file_name_stub + '/'
    if not Path(results_directory_stub).is_dir():
        os.mkdir(results_directory_stub)

    # fixed parameters
    error_type = 'mlogloss'
    threshold_error = 1  # 0.09 *CHANGE THIS TO WHICHEVER THRESHOLD YOU WANT (default at 1)
    total_number_of_iterations = 100
    number_of_random_points = 5
    number_of_iterations = total_number_of_iterations - number_of_random_points
    save_models = False
    num_models_to_accept = 5

    # read data
    x_calib = np.load(data_directory + 'screenshots.npy')
    y_calib = np.load(data_directory + 'command_keys.npy')

    # Comment out if already converted.
    y_calib[y_calib == -1] = 0
    y_calib[y_calib == 38] = 1

    print(np.unique(y_calib, return_counts=True))
    d_calib = xgb.DMatrix(x_calib, label=y_calib)
    num_classes = np.unique(y_calib).shape[0]
    x_train, x_valid, y_train, y_valid = train_test_split(x_calib, y_calib,
                                                          train_size=0.75,
                                                          shuffle=True,
                                                          stratify=y_calib)

    # transform numpy arrays into dmatrix format
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    print('\n*** starting at', pd.Timestamp.now())

    if calculation_type == 'calibration':

        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)

        create_save_models(dtrain=d_train,
                           dvalid=d_valid,
                           error_metric=error_type,
                           num_random_points=number_of_random_points,
                           num_iterations=number_of_iterations,
                           results_dir=results_directory,
                           processor_type=type_of_processor,
                           number_of_classes=num_classes)

    elif calculation_type == 'production':

        # get etrees parameters
        models_dir = results_directory_stub + 'calibration/'
        df_parameters = pd.read_pickle(models_dir + 'df_results.pkl')

        dict_file = open(models_dir + 'dict_single_parameters.pkl', 'rb')
        dictionary_single_parameters = pickle.load(dict_file)

        results_directory = results_directory_stub + calculation_type + '/'
        if not Path(results_directory).is_dir():
            os.mkdir(results_directory)

        x_prod = np.load(data_directory + 'screenshots.npy')
        y_prod = np.load(data_directory + 'command_keys.npy')

        # Comment out if already converted.
        y_prod[y_prod == -1] = 0
        y_prod[y_prod == 38] = 1

        num_classes = np.unique(y_prod).shape[0]
        class_names_list = []
        for i in range(num_classes):
            class_names_list.append('class ' + str(i))

        d_calib = xgb.DMatrix(x_calib, label=y_calib)
        d_prod = xgb.DMatrix(x_prod)
        make_final_predictions(dcalib=d_calib,
                               dprod=d_prod,
                               yprod=y_prod,
                               list_class_names=class_names_list,
                               models_directory=models_dir,
                               save_directory=results_directory,
                               save_models_flag=save_models,
                               df_params=df_parameters,
                               threshold=threshold_error,
                               ml_name=ml_algorithm_name,
                               dict_single_params=dictionary_single_parameters,
                               type_error=error_type,
                               accepted_models_num=num_models_to_accept)

    else:
        print('\ninvalid calculation type:', calculation_type)
        raise NameError
