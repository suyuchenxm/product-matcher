import os
from pathlib import Path
import itertools
from typing import List, Optional
import pickle
import pandas as pd

from hydra import compose, initialize_config_module
from hydra.main import DictConfig

from tensorflow.keras.callbacks import Callback
import time

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from hydra.utils import instantiate


def get_train(X_train, Y_train, training_size):
    return X_train[:training_size], Y_train[:training_size]

def get_path(target_path, model_name, hyperparms=None):
    return Path(target_path, '_'.join([model_name] + [f"{key}_{val}" for key, val in hyperparms.items() if hyperparms is not None]))

def get_hyperparms(**kwargs):
    params = list(kwargs.values())
    return list(itertools.product(*params))

def get_config(overrides: Optional[List[str]] = None) -> DictConfig:
    """Convenience function that returns an hydra config.
    Args:
        overrides: a list of overrides for the hydra config.
    """
    overrides = overrides or []
    with initialize_config_module(config_module="conf", version_base=None):
        return compose(config_name="config", overrides=overrides)

def loader(file, format):
    if format == 'pickle':
        return pickle.load(open(file, 'rb'))
    if format == 'csv':
        return pd.read_csv(file)

def load_data(cfg, PROJECT_PATH):
    train = loader(
        os.path.join(
            PROJECT_PATH,
            cfg['artifacts']['training_data']['name']
            ),
        cfg['artifacts']['training_data']['format']
        )
    test = loader(
        os.path.join(
            PROJECT_PATH,
            cfg['artifacts']['testing_data']['name']
            ),
        cfg['artifacts']['testing_data']['format']
        )

    X_train = loader(
        os.path.join(
            PROJECT_PATH,
            cfg['artifacts']['training_features']['name']
            ),
        cfg['artifacts']['training_features']['format']
        )
    Y_train = train['label'].values

    vectorizer = loader(
        os.path.join(
            PROJECT_PATH,
            cfg['artifacts']['tfidf_vectorizer']['name']
            ),
        cfg['artifacts']['tfidf_vectorizer']['format']
        )

    X_test = vectorizer.transform(test['TITLE_CLEANED'])
    Y_test = test['label'].values

    return train, test, X_train, Y_train, X_test, Y_test

class timecallback(Callback):
    def __init___(self):
        self.predict_time = []
    def on_train_begin(self, logs={}):
        self._start_time = time.time()
    def on_train_end(self, logs={}):
        self._end_time = time.time()
        self.training_time = self._end_time - self._start_time


def get_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def create_nn_experiments(param, n_features, seed):
    experiments1, experiments2 = [], []

    for activation in param['activation']:
        layers = [Dense(units=unit, activation=activation) for unit in param['units']]
        models1 = [
            tf.keras.Sequential(
                [tf.keras.layers.InputLayer((n_features,)),
                 layer,
                 tf.keras.layers.Dense(1)]
                )
            for layer in layers
            ]
        models2 = [
            tf.keras.Sequential(
                [tf.keras.layers.InputLayer((n_features,)),
                 layer,
                 Dropout(0.2, seed=seed),
                 tf.keras.layers.Dense(1)]
                )
            for layer in layers
            ]
        experiments1.append(
            {"layer": "dense", "activation": activation, "models": models1,
             "model_units": param["units"]}
            )
        experiments2.append(
            {"layer": "dense_dropout", "activation": activation, "models": models2,
             "model_units": param["units"]}
            )
    return experiments1 + experiments2
