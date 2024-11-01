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
    rmse = metrics.root_mean_squared_error(actual, predicted)
    r2 = metrics.r2_score(actual, predicted)

    return {'rmse': rmse, 'r2': r2}

def retrieve_best(grid_object):
    best_model = grid_object.best_estimator_    
    best_params = grid_object.best_params_
    best_cv_score = grid_object.best_score_
    return best_model, best_params, best_cv_score

def get_cv_scores(grid_search_results):
    best_score_index = np.where(grid_search_results['rank_test_neg_root_mean_squared_error'] == 1)[0]
    splits_keys = [key for key in grid_search_results.keys() if 'split' in key]
    cv_scores = {'test_r2': [], 'test_neg_root_mean_squared_error': []}
    for key in splits_keys:
        if 'r2' in key:
            cv_scores['test_r2'].append(grid_search_results[key][best_score_index])
        elif 'neg_root_mean_squared_error' in key:
            cv_scores['test_neg_root_mean_squared_error'].append(grid_search_results[key][best_score_index])
    for key, value in cv_scores.items():
        cv_scores[key] = np.array(value)
    return cv_scores


def predict_model(model, datasets):
    results_dict = {}
    for type, dataset in datasets.items():
        prediction_array = model.predict(dataset)
        prediction_series = pd.Series(np.ravel(prediction_array), index = dataset.index, name = 'PredShare')
        results_dict[type] = prediction_series
    return results_dict

def log_sampling_ratio_mlflow(ratio):
    mlflow.log_param('sampling_ratio', ratio)

def log_params_mlflow_enet(params):
    mlflow.log_param('alpha', params['alpha'])
    mlflow.log_param('l1_ratio', params['l1_ratio'])

def log_params_mlflow_xgb(params):
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('colsample_bytree', params['colsample_bytree'])
    mlflow.log_param('learning_rate', params['learning_rate'])
    mlflow.log_param('n_estimators', params['n_estimators'])
    mlflow.log_param('subsample', params['subsample'])

def log_params_mlflow_dt(params):
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('max_features', params['max_features'])
    mlflow.log_param('min_samples_split', params['min_samples_split'])

def log_params_mlflow_rf(params):
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('max_features', params['max_features'])
    mlflow.log_param('min_samples_split', params['min_samples_split'])
    mlflow.log_param('n_estimators', params['n_estimators'])

def log_params_mlflow_ens(params):
    weights = params['weights']
    if weights is None:
        weight_xgb = 1
        weight_rf = 1
    else:
        weight_xgb = weights[0]
        weight_rf = weights[1]
    mlflow.log_param('weight_xgb', weight_xgb)
    mlflow.log_param('weight_rf', weight_rf)

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

    rmse_contenders = metrics.root_mean_squared_error(results_contenders['Share'], results_contenders['PredShare'])
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
    graph.set_xticklabels(graph.get_xticklabels(),rotation = 30)
    return graph

def display_feature_importances(model):
    coef_df = pd.DataFrame(model.feature_importances_.T, index = model.feature_names_in_, columns = ['coef'])
    coef_df['abs_coef'] = abs(coef_df['coef'])
    coef_df_top = coef_df.sort_values('abs_coef', ascending = False).nlargest(10, 'abs_coef')
    graph = sns.barplot(x = coef_df_top.index, y = coef_df_top['coef'], edgecolor = 'black')
    graph.set_title('Most important features')
    graph.set_ylabel('Importance coefficient')
    graph.set_xlabel('Feature')
    graph.set_xticklabels(graph.get_xticklabels(),rotation = 30)
    return graph  

def display_feature_importances_xgb(model):
    coef_df = pd.DataFrame(model.feature_importances_.T, index = model.get_booster().feature_names, columns = ['coef'])
    coef_df['abs_coef'] = abs(coef_df['coef'])
    coef_df_top = coef_df.sort_values('abs_coef', ascending = False).nlargest(10, 'abs_coef')
    graph = sns.barplot(x = coef_df_top.index, y = coef_df_top['coef'], edgecolor = 'black')
    graph.set_title('Most important features')
    graph.set_ylabel('Importance coefficient')
    graph.set_xlabel('Feature')
    graph.set_xticklabels(graph.get_xticklabels(),rotation = 30)
    return graph  

def log_important_features_mlflow(graph):
    png_file_path = os.path.join(os.getcwd(), 'plots/important_features.png')
    fig = graph.get_figure()
    plt.close(fig)
    fig.savefig(png_file_path)
    mlflow.log_artifact(png_file_path)

def display_val_results_graphs(real_val_y, pred_val_y, players_series):
    contenders_df, no_contenders_df = get_val_results(real_val_y, pred_val_y, players_series)
    seasons = set(contenders_df.index.get_level_values(1))
    
    fig_cont, axs_cont = plt.subplots(int(len(seasons) // 2 + len(seasons) % 2), 2 , figsize = (40, 40));
    fig_cont.subplots_adjust(hspace=.4)
    for counter, season in enumerate(seasons):
        cont_season_df = contenders_df.loc[pd.IndexSlice[:, season], :].sort_values(by = 'Share', ascending = False)
        cont_season_df.rename({'Share': 'Real Share', 'PredShare': 'Predicted Share'}, axis = 1, inplace = True)
        cont_season_df = cont_season_df.melt(id_vars = ['Player'], value_vars = ['Real Share', 'Predicted Share'], var_name = 'Type', value_name = 'Value')
        sns.barplot(ax = axs_cont.flat[counter], x = 'Player', y = 'Value', hue = 'Type', data = cont_season_df)
        axs_cont.flat[counter].set_title(f'MVP Voting Share for Contenders in {season - 1} - {season}')
        axs_cont.flat[counter].set_ylabel('Voting Share Value')
        axs_cont.flat[counter].set_xticklabels(axs_cont.flat[counter].get_xticklabels(),rotation = 30)
        axs_cont.flat[counter].set_ylim((0, 1))
    
    fig_no_cont, axs_no_cont = plt.subplots(int(len(seasons) // 2 + len(seasons) % 2), 2 , figsize = (40, 40));
    fig_no_cont.subplots_adjust(hspace=.4)
    for counter, season in enumerate(seasons):
        no_cont_season_df = no_contenders_df.loc[pd.IndexSlice[:, season], :].sort_values(by = 'PredShare', ascending = False)
        no_cont_season_df = no_cont_season_df[no_cont_season_df['PredShare'] > .1].nlargest(15, 'PredShare')
        if len(no_cont_season_df) > 0:
            sns.barplot(ax = axs_no_cont.flat[counter], x = 'Player', y = 'PredShare', data = no_cont_season_df)
            axs_no_cont.flat[counter].set_title(f'Predicted MVP Voting Share for No Contenders in {season - 1} - {season} Season')
            axs_no_cont.flat[counter].set_ylabel('Predicted Voting Share Value')
            axs_no_cont.flat[counter].set_xticklabels(axs_no_cont.flat[counter].get_xticklabels(),rotation = 30)
            axs_no_cont.flat[counter].set_ylim((0, 1))

    return fig_cont, fig_no_cont

def log_val_results_mlflow(graph_cont, graph_no_cont):
    png_file_path_cont = os.path.join(os.getcwd(), 'plots/contenders_val_results.png')
    png_file_path_no_cont = os.path.join(os.getcwd(), 'plots/no_contenders_val_results.png')
    fig_cont = graph_cont.get_figure()
    plt.close(fig_cont)
    fig_cont.savefig(png_file_path_cont)
    mlflow.log_artifact(png_file_path_cont)
    fig_no_cont = graph_no_cont.get_figure()
    plt.close(fig_no_cont)
    fig_no_cont.savefig(png_file_path_no_cont)
    mlflow.log_artifact(png_file_path_no_cont)