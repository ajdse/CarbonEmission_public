import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Sklearn Imports
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # <-- Added this line
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor 
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

VALID_CONFIG_KEYS = [
    'StandardScaler_config',
    'OneHotEncode_config',
    'Tfidf_config',
    # Add other keys here as you expand
    'remainder',
    'sparse_threshold',
    'final_feature_subset'
]

class IndustryStatsTransformer(BaseEstimator, TransformerMixin):
    """Calculates Mean and Std Dev per industry ONLY on fit data."""
    def __init__(self, industry_col_name='industrycode',smooth_param=0.8):
        self.industry_col_name = industry_col_name
        self.smooth_param = smooth_param
        self.industry_mean_map_ = {}
        self.industry_std_map_ = {}
        self.global_mean_ = 0
        self.global_std_ = 0

    def smoothe_mean(self,mean, count, global_mean):
        lambda_smooth = count/(count + self.smooth_param)
        smoothed_mean = lambda_smooth * mean + (1 - lambda_smooth) * global_mean
        return smoothed_mean

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """"
        Docstring
        """
        if y is None: raise ValueError("y must be provided.")
        if X is None: raise ValueError("a dataframe containing industry col name")
        train_df = X.copy()
        train_df['_outcome_'] = y

        self.global_mean_ = train_df['_outcome_'].mean()
        self.global_std_ = train_df['_outcome_'].std()

        # Calculate mean and std dev dynamically from this fold's training data
        industry_summary_stats = train_df.groupby(self.industry_col_name)['_outcome_'].agg([['mean','count','var']])
        industry_summary_stats['smoothed_mean'] = industry_summary_stats.apply(
            lambda row: self.smoothe_mean(row['mean'], row['count'], self.global_mean_), axis=1)

        # if the industry is only represented by one company in our dataset, the standard deviation is treated as zero
        self.industry_std_map_ = train_df.groupby(self.industry_col_name)['_outcome_'].agg('std').fillna(0).to_dict()


        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['Industry_Current_Mean'] = X_transformed[self.industry_col_name].map(self.industry_mean_map_).fillna(self.global_mean_)
        X_transformed['Industry_Current_StdDev'] = X_transformed[self.industry_col_name].map(self.industry_std_map_).fillna(self.global_std_)
        
        # Return only the new features as a DataFrame
        return X_transformed[['Industry_Current_Mean', 'Industry_Current_StdDev']]

class WinsorizeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                    lower_percentile=0.1,
                    upper_percentile=0.99,
                    only_upper=True):
        
        self.lower_per = lower_percentile
        self.upper_per = upper_percentile
        self.only_upper = only_upper
        self.lower_clip_val = None
        self.upper_clip_val = None

        
    def fit(self, Series:pd.Series):
        """Fit the winsorizer for the series using the percentiles
        specified.
        Args:
            Series: pd.Series to fit the winsorizer on
    
        """
        self.upper_clip_val = Series.quantile(self.upper_per)
        if self.only_upper == True:
            self.lower_clip_val = Series.min()
    
    def transform(self, Series:pd.Series):
        """ Transform the series by winsorizing it
        Args:
            Series: pd.Series to transform
        Returns:
            pd.Series with winsorization applied
        """
        return Series.clip(lower=self.lower_clip_val, upper=self.upper_clip_val)

def clean_text(Xseries):

    # 1. fill na since there could be missing values
    Xseries = Xseries.fillna(' ')

    # 2. make str lower
    Xseries = Xseries.str.lower()

    # 3. Remove Punctuation and Digits (Adjust regex based on what you need to keep)
    # This keeps only letters (a-z) and spaces (\s)

    Xseries = Xseries.apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))
    
    # 4. Normalize Whitespace (remove multiple spaces and trim edges)
    Outseries = Xseries.str.strip().str.replace(r'\s+', ' ', regex=True)
    
    return Outseries
    
def build_datapreprocessor_pipeline(**kwargs):
    """ Build the data preprocessor pipeline based on provided configurations
    returns: sklearn Pipeline object

    The job of this function is to dynamically create a ColumnTransformer
    based on the configurations provided in kwargs
    """

    # extract the configs
    standard_scaler_config = kwargs.get('StandardScaler_config')
    one_hot_encode_config = kwargs.get('OneHotEncode_config')
    tfidf_embed_encode_config = kwargs.get('Tfidf_config')

    # prepare the ColumnTransformer
    processor_list = []
    
    if standard_scaler_config:
        numeric_features_to_scale = standard_scaler_config.get('features')
        processor_list.append(('standardscaler',StandardScaler(), numeric_features_to_scale))
        
    if one_hot_encode_config:
        handleUnk = one_hot_encode_config.get('handle_unknown','ignore')
        categorical_features = one_hot_encode_config.get('features')
        processor_list.append(('onehotencoder',OneHotEncoder(handle_unknown=handleUnk),
                               categorical_features))
            
    if tfidf_embed_encode_config:
        max_features = tfidf_embed_encode_config.get('max_features',100)
        ngram_range = tfidf_embed_encode_config.get('ngram_range',(1,2))
        text_feature_name = tfidf_embed_encode_config.get('features')
        processor_list.append(
            ('tfidfvectorizer',
            TfidfVectorizer(max_features=max_features,
                            ngram_range=ngram_range,
                            stop_words='english',
                            token_pattern=r'\b[a-zA-Z]{3,}\b'),
            text_feature_name)
            )
    
    preprocessor = ColumnTransformer(
    transformers=processor_list, # Pass the list directly to the 'transformers' kwarg
    remainder=kwargs.get('remainder', 'passthrough'),
    sparse_threshold=kwargs.get('sparse_threshold', 0))
        
    " prepare the model pipeline"
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
    #    ('regressor', XGBRegressor(objective='reg:squarederror'))
    ])

    return pipeline

def build_model_pipeline(model_config):
    """ Generate the model pipeline with a model
    args:
        model_config: dict containing model hyperparameters for XGBRegressor
    returns: sklearn model pipeline
    """
    # Train the model
    objective = model_config.get('objective', 'reg:squarederror')

    model_pipeline = Pipeline(
        steps=[('XGBregressor', XGBRegressor(objective=objective))])
    
    return model_pipeline

def aug_data(X_train, y_train, X_val, industry_col_name='industrycode'):
    """ Augument the data with industry stats or other transformations.
    args:
        X_train: Training features dataframe
        X_val: Validation features dataframe
    returns: Processed training and validation feature dataframes
    """      
    stats_transformer = IndustryStatsTransformer(industry_col_name=industry_col_name)

    stats_transformer.fit(X_train, y_train)

    X_train_stats = stats_transformer.transform(X_train)

    X_val_stats = stats_transformer.transform(X_val)

    X_train_aug = pd.concat([X_train, X_train_stats], axis=1)

    X_val_aug = pd.concat([X_val, X_val_stats], axis=1)

    return X_train_aug, X_val_aug

def process_data(
        feature_config,
        X_train,
        X_val):
    """ Process the training & validation data based on feature configuration and industry stats
    args:
        feature_config: dict containing feature processing configurations
        X_train: Training features dataframe
        X_val: Validation features dataframe
    returns: Processed training and validation feature dataframes
    """      
    X_train_cln = filter_unembedded_textfeatures(feature_config, X_train)
    X_val_cln = filter_unembedded_textfeatures(feature_config, X_val)

    data_processor = build_datapreprocessor_pipeline(**feature_config)
    
    X_train_processed = data_processor.fit_transform(X_train_cln)
    X_val_processed = data_processor.transform(X_val_cln)

    # --- EXTRACT FEATURE NAMES HERE ---
    feature_names_out = data_processor.named_steps['preprocessor'].get_feature_names_out()


    return X_train_processed,X_val_processed,feature_names_out

def fit_eval_model_pipeline(X_train_processed,
                            y_train,
                            X_val_processed,
                            y_val,
                            model_pipeline=None, model_config={}):
    """ Train and evaluate the  model pipeline. the pipeline could
    args:
        model_pipeline: sklearn Pipeline object with model
        X_train_processed: Processed training features
        y_train: Training target variable
        X_val_processed: Processed validation features
        y_val: Validation target variable
    returns: Trained model pipeline, validation predictions
    """
    model_pipeline = build_model_pipeline(model_config)
    fitted_model = model_pipeline.fit(X_train_processed, y_train)
    r2 = fitted_model.score(X_val_processed, y_val)
    val_predictions = fitted_model.predict(X_val_processed)
    fe = fitted_model.named_steps['XGBregressor'].feature_importances_
    
    return fitted_model, val_predictions, r2, fe


def prepare_model1_data(full_data,
                        latest_year,
                        target_col,
                        company_id_col,
                        text_feature_name,
                        test_size_per=0.20
                        ):
    """ Returns a Tuple of pandas dataframes

    (Training&Validation dataset for regressors,
    Training&Validation dataset for target variable,
    groups that are part of training&Validation dataset,
    Test data set for regressors,
    Test data set for target variable)

    """

    # obtain the latest data based on latest year
    # we expect only one text feature name rather than a list of features requiring embedding
    full_data.loc[:,[text_feature_name]] = clean_text(full_data[text_feature_name])

    model1_full_snapshot_data = full_data[full_data['year'] == latest_year].copy()


    # now drop target column, id col and year from the maindata to arrive 
    X_m1_full = model1_full_snapshot_data.drop([target_col, company_id_col, 'year'], axis=1)
    y_m1_full = model1_full_snapshot_data[target_col]
    groups_m1_full = model1_full_snapshot_data[company_id_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size_per, random_state=42)
    train_val_idx, test_idx = next(gss.split(X_m1_full, y_m1_full, groups_m1_full))

    X_train_val = X_m1_full.iloc[train_val_idx].reset_index(drop=True)
    y_train_val = y_m1_full.iloc[train_val_idx].reset_index(drop=True)
    groups_train_val = groups_m1_full.iloc[train_val_idx].reset_index(drop=True)

    X_test = X_m1_full.iloc[test_idx].reset_index(drop=True)
    y_test = y_m1_full.iloc[test_idx].reset_index(drop=True)

    return X_train_val, y_train_val, groups_train_val, X_test, y_test

def validate_config(config_dict):
    """
    Checks the provided configuration dictionary for validity before running the pipeline.
    """
    # Check for unrecognized keys
    for key in config_dict.keys():
        if key not in VALID_CONFIG_KEYS:
            raise ValueError(f"Unknown configuration key found: '{key}'. Check spelling. Valid keys are: {VALID_CONFIG_KEYS}")
            
    # Add more specific checks here later (e.g., ensure 'features' is always present in sub-dicts)
    for key, sub_config in config_dict.items():
        if key.endswith('_config') and 'features' not in sub_config:
             raise ValueError(f"Configuration '{key}' is missing the required 'features' key.")
             
    print("Configuration validated successfully.")
    return True

def filter_unembedded_textfeatures(config_dict, X_full_df):
    """
    Subsets the DataFrame X_full_df to include only the features specified 
    in the config_dict dictionary across all transformer configurations.
    """
    
    # 1. Gather all required features from the config dictionary

    dtypes = X_full_df.dtypes
    text_feat_set_in_df = set(dtypes[dtypes == 'object'].index)
    text_embedders = ['OneHotEncode_config','Tfidf_config']

    # Check all possible configuration keys
    text_feautres_embedded = []
    for config_key in config_dict.keys():
        if config_key in text_embedders:
            text_features = config_dict.get(config_key).get('features')
            if isinstance(text_features, list):
                text_feautres_embedded = text_feautres_embedded + text_features
            else:
                text_feautres_embedded.append(text_features)
        
    remaining_text_features = text_feat_set_in_df - set(text_feautres_embedded)

    X_subset = X_full_df.drop(list(remaining_text_features), axis=1)
            
    return X_subset

def run_model_scope1_v1():
    """ Run the model for scope 1 emissions - version 1
    This version models the absolute emissions as a ratio against industry mean
    and also as a raw value with industry stats as features
    """
    pass

def generate_kfold_data (
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        groups_train_val: pd.Series,
        feature_transform_config: dict,
        nsplits=2,
        industry_col_name='industrycode',
        obtain_industry_stats=True,
        **kwargs):
    """ Generate K-Fold data splits with processed features
    args:
        X_train_val: Training & Validation features dataframe
        y_train_val: Training & Validation target variable series
        groups_train_val: Group labels for the training & validation data
        feature_transform_config: Configuration dictionary for feature processing
        nsplits: Number of splits for K-Fold cross-validation
        obtain_industry_stats: Whether to augment data with industry statistics
        industry_col_name: Column name for industry codes defaults to 'industrycode' 
        kwargs: Additional arguments for processing (e.g., outcome winsorization settings)
    returns: Dictionary with fold indices as keys and processed data as values
    
    The processed data for each fold includes:
        - 'X_train': Processed training features (numpy array)
        - 'X_val': Processed validation features (numpy array)
        - 'feature_names_out': List of feature names after processing
        - 'X_train_aug': Augmented training features (pandas DataFrame)
        - 'X_val_aug': Augmented validation features (pandas DataFrame)
        - 'y_train': Training target (dependent on winsorization) variable (pandas Series)
        - 'y_val': Validation target (dependent on winsorization) variable (pandas Series)
    """

    validate_config(feature_transform_config)
    data_out = {}
    outcome_winsorize_setting = kwargs.get('outcome_winsorize_setting',None)

    ## 3.2 Run Cross-Validation to evaluate model performance
    group_kfold = GroupKFold(n_splits=nsplits)

    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(X_train_val, y_train_val, groups_train_val)):

        # Manual data splitting and dynamic feature addition (still required for leakage prevention)
        X_train_raw, X_val_raw = X_train_val.iloc[train_idx].reset_index(drop=True), X_train_val.iloc[val_idx].reset_index(drop=True)
        y_train_raw, y_val_raw = y_train_val.iloc[train_idx].reset_index(drop=True), y_train_val.iloc[val_idx].reset_index(drop=True)
        
        # Winsorize the outcome variable if required
        if outcome_winsorize_setting is not None:
            lower_per = outcome_winsorize_setting.get('lower_per', 0.01)
            upper_per = outcome_winsorize_setting.get('upper_per', 0.99)
            only_upper = outcome_winsorize_setting.get('only_upper', True)
               
            winsorizer = WinsorizeTransformer(lower_per, upper_per, only_upper=only_upper)
            winsorizer.fit(y_train_raw)
            y_train_raw = winsorizer.transform(y_train_raw)
            y_val_raw = winsorizer.transform(y_val_raw)
    
        # Augment data with industry stats if required
        
        if obtain_industry_stats:
            X_train_aug, X_val_aug = aug_data(X_train_raw, y_train_raw, X_val_raw,
                                              industry_col_name=industry_col_name)
        else:
            X_train_aug, X_val_aug = X_train_raw, X_val_raw

        # Process and augment data using original y val the outputs are arrays and not dataframes/series
        X_train_processed, X_val_processed, feature_names_out = process_data(
            feature_transform_config,
            X_train_aug,
            X_val_aug)
        
        # X_train_processed and X_val_processed are now ready for modeling. these are arrays
        data_out[fold_idx] = {
            'X_train_processed': X_train_processed,
            'X_val_processed': X_val_processed,
            'feature_names_out': feature_names_out,
            'X_train_aug': X_train_aug,
            'X_val_aug': X_val_aug,
            'y_train': y_train_raw,
            'y_val': y_val_raw
        }

    return data_out

if __name__ == "__main__":
    pass