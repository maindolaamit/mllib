import importlib
import os

from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img
from tensorflow.keras.layers import Dense, Dropout

MODELS_FILE_DIR = Path(__file__).resolve().parent
MODELS_JSON_FILE_PATH = os.path.join(MODELS_FILE_DIR, 'models.json')


class CNNModel:
    def __init__(self, base_model, weights='imagenet', input_shape=(224, 224, 3), loss='categorical_crossentropy'):
        """
        Constructor method
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        self.base_model = base_model
        self.weights = weights
        self.input_shape = input_shape
        self.model = None
        self.loss = loss

    def _get_base_module(self, model_name):
        """
        Get the base model based on the base model name
        :param model_name: Base model name
        :return: Base models' library
        """
        import json
        with open(MODELS_JSON_FILE_PATH) as model_json_file:
            models = json.load(model_json_file)
        if model_name not in models.keys():
            raise Exception(f"Invalid model name, should have one of the value {models.keys()}")
        self.base_model_name = models[model_name]['model_name']
        model_package = models[model_name]['model_package']
        print(f"{model_package}.{self.base_model_name}")
        self.base_module = importlib.import_module(model_package)

    def build(self):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.base_model_name)
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.weights,
                                   pooling='avg', include_top=False)
        self.model = Dropout(0.5)(self.base_model)
        self.model = Dense(10, activation='softmax')(self.model)
        return self.model

    def compile(self):
        """
        Compile the Model
        """
        self.model.compile(optimizer=Adam(), loss=self.loss)

    def get_base_model_attrs(self):
        preprocess_input = getattr(self.base_module, 'preprocess_input')
        decode_predictions = getattr(self.base_module, 'decode_predictions')
        predict = getattr(self.base_model, 'predict')
        return preprocess_input, decode_predictions, predict
