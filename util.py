
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import numpy as np


def get_train_test_dfs(dataframe,test_size):
        X_df = dataframe.drop('target', axis = 'columns')
        y_df = dataframe['target']
        #y = self.dataframe['target'].values
        # Perform a stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_size, stratify=y_df)
        
        # Combine X and y into single DataFrames
        train_df = X_train.copy()
        train_df['target'] = y_train.values
        
        val_df = X_test.copy()
        val_df['target'] = y_test.values

        return train_df, val_df

def get_sampler(df):
        sample_weights = np.zeros(shape = (len(df)))
        vc = df['target'].value_counts()
        sample_weights[df['target'] == 1] = (1 / vc[1])
        sample_weights[df['target'] == 0] = (1 / vc[0])
        sampler = WeightedRandomSampler(weights = sample_weights, num_samples=len(sample_weights), replacement=False)
        return sampler
    
        