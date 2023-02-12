from sklearn.model_selection import GridSearchCV
from sklearn import tree
from product_matcher.utils import *
SEED = 123
PROJECT_PATH = ".."
import hydra

@hydra.main(version_base=None)
def run(cfg: DictConfig):
    cfg = get_config(overrides=['experiments=problem1/decisiontree'])

    train, test, X_train, Y_train, X_test, Y_test = load_data(cfg, PROJECT_PATH)

    param_grid = {
        'max_depth': cfg['experiments']['model']['max_depth'],
        'criterion': cfg['experiments']['model']['criterion']
    }
    TRAINING_SIZE = cfg['training_sizes']['problem1']

    cv_results=dict()
    for training_size in TRAINING_SIZE:
        print(f"training size: {training_size}")
        x,y = get_train(X_train, Y_train, training_size=training_size)
        clf = tree.DecisionTreeClassifier()
        gscv = GridSearchCV(clf,
                            param_grid=param_grid,
                            scoring=['accuracy', 'f1', 'recall', 'precision'],
                            refit=False,
                            cv=10,
                            return_train_score=True)
        gscv.fit(x, y)
        _report = gscv.cv_results_
        _report['training_size'] = training_size
        cv_results[training_size] = _report

        pickle.dump(
            cv_results, open(
                os.path.join(
                    PROJECT_PATH, cfg['artifacts']['hyperparameter_study']['name']
                    ), "wb"
                )
            )

if __name__ == "__main__":
    run()