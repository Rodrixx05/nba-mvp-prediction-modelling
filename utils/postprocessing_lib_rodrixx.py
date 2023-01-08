import numpy as np
import pandas as pd


def get_processed_prediction(pred_val_y, players_names, num_contenders = 17, total_share = 2.6, max_votes = 1010): 
    df_results = pd.concat([players_names, pred_val_y], axis = 1)
    df_results.sort_values('PredShare', ascending = False, inplace = True)
    df_results['PredShare_Adj'] = 0
    df_results.iloc[:num_contenders, 2] = df_results.iloc[:num_contenders, 1]
    sum_share = df_results['PredShare_Adj'].sum()
    df_results['PredShare_Adj'] = df_results['PredShare_Adj'] * total_share / sum_share
    df_results['PredVotes'] = (df_results['PredShare_Adj'] * max_votes).round(0).astype(int)
    df_results['PredShare_Adj'] = (df_results['PredVotes'] / max_votes).round(3)
    df_results['PredRank'] = df_results['PredShare'].rank(method = 'min', ascending = False).astype(int)
    return df_results




