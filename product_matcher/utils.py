import os
from pathlib import Path
import itertools
from typing import List, Optional
import pickle
import pandas as pd

from hydra import compose, initialize_config_module
from hydra.main import DictConfig


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
