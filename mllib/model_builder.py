import os
import importlib
from pathlib import Path
from sklearn import model_selection

from utils.file_utils import load_from_json
from utils.misc import get_display_time

MODELS_FILE_DIR = Path(__file__).resolve().parent
CLF_MODELS_JSON_FILE = os.path.join(MODELS_FILE_DIR, 'clf_models.json')
REG_MODELS_JSON_FILE = os.path.join(MODELS_FILE_DIR, 'reg_models.json')


class ClassifierModel:
    def __init__(self, model_name, **model_kwargs):
        self.model_name = None
        self.model = None
        self.param_grid = None
        self.best_estimator = None
        self._set_model(model_name, **model_kwargs)

    def _set_model(self, model_name, **model_kwargs):
        models = load_from_json(CLF_MODELS_JSON_FILE)
        if model_name not in models.keys():
            raise Exception(f"Model name should have one of the value {models.keys()}")
        self.model_name = models[model_name]['model_name']
        self.param_grid = models[model_name]['param_grid']
        model_package = models[model_name]['model_package']
        print(f"{model_package}.{self.model_name}")
        module = importlib.import_module(model_package)
        model_class = getattr(module, self.model_name)
        self.model = model_class(**model_kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def fine_tune(self, X, y, param_grid=None, cv=5, randomized=False, verbose=0):
        from time import perf_counter
        start_time = perf_counter()

        if param_grid is not None:
            self.param_grid = param_grid

        grid_search = None
        if randomized:
            print(f"Performing Randomized search for {type(self.model).__name__}...")
            grid_search = model_selection.RandomizedSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
        else:
            print(f"Performing Grid search for {type(self.model).__name__}...")
            grid_search = model_selection.GridSearchCV(self.model, param_grid, cv=cv, verbose=verbose, n_jobs=-1)

        # Start fine tuning of the model
        grid_search.fit(X, y)
        time_taken = round(perf_counter() - start_time, 2)
        print(f"Time elapsed : {get_display_time(time_taken)} | score : {grid_search.best_score_:.2}")
        print(f"Best parameters : {grid_search.best_params_} ")
        self.best_estimator = grid_search.best_estimator_
        return grid_search.best_estimator_
