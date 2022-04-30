import pandas as pd
import numpy as np
import os
import csv

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn import metrics

from imblearn.over_sampling import RandomOverSampler, SMOTENC

import mlflow

from IPython.display import display

def oversample(df, os_param = 'ros', sampling_strategy = .25):
    df_os = df.copy()
    df_os.loc[:, 'Contender'] = (df_os['Share'] > 0) * 1
    df_X = df_os.drop('Contender', axis = 1)
    df_y = df_os[['Contender']]

    if os_param == 'smote':
        int_cols = df_X.select_dtypes('int').columns
        cat_index = [df_X.columns.get_loc(column) for column in int_cols]
        os_technique = SMOTENC(sampling_strategy = sampling_strategy, random_state=23, categorical_features= cat_index)  
    else:
        os_technique = RandomOverSampler(sampling_strategy = sampling_strategy, random_state = 23)

    X_resampled, y_resampled = os_technique.fit_resample(df_X, df_y)
    df_ros = pd.concat([X_resampled, y_resampled], axis = 1)
    df_ros.drop('Contender', axis = 1, inplace = True)

    return df_ros, sampling_strategy

def eval_metrics(actual, predicted):
    rmse = np.sqrt(metrics.mean_squared_error(actual, predicted))
    r2 = metrics.r2_score(actual, predicted)

    return {'rmse': rmse, 'r2': r2}

def retrieve_best(grid_object):
    best_model = grid_object.best_estimator_    
    best_params = grid_object.best_params_
    best_cv_score = grid_object.best_score_
    best_params['best_ntree_limit'] = best_model.best_ntree_limit
    return best_model, best_params, best_cv_score

def predict_model(model, datasets):
    results_dict = {}
    for type, dataset in datasets.items():
        prediction_array = model.predict(dataset)
        prediction_series = pd.Series(np.ravel(prediction_array), index = dataset.index, name = 'PredShare')
        results_dict[type] = prediction_series
    return results_dict

def log_params_mlflow_xgb(params, sampling_ratio):
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('colsample_bytree', params['colsample_bytree'])
    mlflow.log_param('learning_rate', params['learning_rate'])
    mlflow.log_param('n_estimators', params['n_estimators'])
    mlflow.log_param('best_ntree_limit', params['best_ntree_limit'])
    mlflow.log_param('subsample', params['subsample'])
    mlflow.log_param('sampling_ratio', sampling_ratio)

def log_params_mlflow_rf(params, sampling_ratio):
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('max_features', params['max_features'])
    mlflow.log_param('min_samples_split', params['min_samples_split'])
    mlflow.log_param('n_estimators', params['model__n_estimators'])
    mlflow.log_param('sampling_ratio', sampling_ratio)

def get_metrics(targets_real, targets_predicted, cv_scores):
    train_metrics = eval_metrics(targets_real['train'], targets_predicted['train'])
    cv_metrics = {'rmse': abs(cv_scores['test_neg_root_mean_squared_error']).mean(), 'r2': cv_scores['test_r2'].mean()}
    val_metrics = eval_metrics(targets_real['val'], targets_predicted['val'])
    return train_metrics, cv_metrics, val_metrics

def log_metrics_mlflow(targets_real, targets_predicted, cv_scores):
    train_metrics, cv_metrics, val_metrics = get_metrics(targets_real, targets_predicted, cv_scores)

    mlflow.log_metric('rmse_train', train_metrics['rmse'])
    mlflow.log_metric('r2_train', train_metrics['r2'])
    mlflow.log_metric('rmse_cv', cv_metrics['rmse'])
    mlflow.log_metric('r2_cv', cv_metrics['r2'])
    mlflow.log_metric('rmse_val', val_metrics['rmse'])
    mlflow.log_metric('r2_val', val_metrics['r2'])

def log_model_mlflow(model):
    mlflow.sklearn.log_model(model, 'model')

def log_df_mlflow(df, path):
    df.to_pickle(os.path.join(path, 'entry_dataframe.pkl'))
    mlflow.log_artifact(os.path.join(path, 'entry_dataframe.pkl'))

def log_features_mlflow(df, path):
    csv_file_path = os.path.join(path, 'features.csv')
    with open(csv_file_path, 'w') as file:
        write = csv.writer(file)
        for column in df.columns:
            write.writerow([column])
    mlflow.log_artifact(csv_file_path)

def display_metrics(targets_real, targets_predicted, cv_scores):
    train_metrics, cv_metrics, val_metrics = get_metrics(targets_real, targets_predicted, cv_scores)
    df_results = pd.DataFrame(
        {
            'Train': [train_metrics['rmse'], train_metrics['r2']],
            'CV': [cv_metrics['rmse'], cv_metrics['r2']], 
            'Validation': [val_metrics['rmse'], val_metrics['r2']],
        }, 
        index = ['RMSE', 'R2']
        )
    
    return df_results

def get_advanced_metrics(y_real, y_predict):

    results = pd.concat([y_real, y_predict], axis = 1)

    results_contenders = results[results['Share'] > 0]
    results_no_contenders = results[results['Share'] == 0]

    rmse_contenders = metrics.mean_squared_error(results_contenders['Share'], results_contenders['PredShare']) ** .5
    mae_no_contenders = metrics.mean_absolute_error(results_no_contenders['Share'], results_no_contenders['PredShare']) ** .5

    return rmse_contenders, mae_no_contenders

def log_advanced_metrics_mlflow(y_real, y_predict):
    rmse_contenders, mae_no_contenders = get_advanced_metrics(y_real, y_predict)
    mlflow.log_metric('rmse_cont', rmse_contenders)
    mlflow.log_metric('mae_no_cont', mae_no_contenders)

def get_val_results(real_val_y, pred_val_y, players_series):

    players_series_val = players_series[players_series.index.get_level_values(1) > 2015]
    results_val = pd.concat([players_series_val, real_val_y, pred_val_y], axis = 1)
    
    results_val_contenders = results_val[results_val['Share'] > 0]
    results_val_no_contenders = results_val[results_val['Share'] == 0]

    return results_val_contenders, results_val_no_contenders

def display_val_results(results_val_contenders, results_val_no_contenders):
    print(f'Contenders Results:')
    for season in set(results_val_contenders.index.get_level_values(1)):
        display(results_val_contenders.loc[pd.IndexSlice[:, season], :].sort_values(by = 'Share', ascending = False))

    print(f'No contenders results:')
    display(results_val_no_contenders[results_val_no_contenders['PredShare'] > 0])

def display_linear_coef(model):
    coef_df = pd.DataFrame(model.coef_.T, index = model.feature_names_in_, columns = ['coef'])
    coef_df['abs_coef'] = abs(coef_df['coef'])
    coef_df_top = coef_df.sort_values('abs_coef', ascending = False).nlargest(10, 'abs_coef')
    graph = sns.barplot(x = coef_df_top.index, y = coef_df_top['coef'], edgecolor = 'black')
    graph.set_title('Most important features')
    graph.set_ylabel('Linear coefficient')
    graph.set_xlabel('Feature')
    return graph

def log_important_features_mlflow(graph):
    png_file_path = os.path.join(os.getcwd(), 'plots/important_features.png')
    fig = graph.get_figure()
    plt.close(fig)
    fig.savefig(png_file_path)
    mlflow.log_artifact(png_file_path)