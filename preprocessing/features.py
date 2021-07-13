import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.impute import SimpleImputer

# Keep randomness same
np.random.seed(2210)


def get_kbest_columns(X, y, k=2, score_func=None):
    """
    Method to fetch columns based on SelectKBest feature selection
    """
    selector = feature_selection.SelectKBest(score_func=score_func, k=k).fit(X, y)
    return X.columns[selector.get_support()]


def get_percentile_columns(X, y, percentile=2, score_func=None):
    """
    Method to fetch columns based on SelectPercentile feature selection
    """
    selector = feature_selection.SelectPercentile(score_func=score_func, percentile=percentile).fit(X, y)
    return X.columns[selector.get_support()]


def get_boruta(X, y):
    """
    Returns the features selected by Boruta algorithm for the passed dataset
    :param X: Numpy array of features
    :param y: Numpy array of target feature
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    # Initialize Boruta
    forest = RandomForestRegressor(
        n_jobs=-1,
        max_depth=5
    )
    boruta = BorutaPy(
        estimator=forest,
        n_estimators='auto',
        max_iter=100  # number of trials to perform
    )
    # fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(np.array(X), np.array(y))
    # print results
    green_area = X.columns[boruta.support_].to_list()
    blue_area = X.columns[boruta.support_weak_].to_list()
    print('features in the green area:', green_area)
    print('features in the blue area:', blue_area)
    print('features ranking :', boruta._rankings)
    return boruta


def get_columns_by_variance(X, y, threshold=0.0):
    """
    Method to fetch columns based on threshold variance
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold).fit(X, y)
    return X.columns[selector.get_support()]


def drop_columns_by_variance(df, threshold=0):
    """
    Method to drop all columns having zero variance
    """
    columns = get_columns_by_variance(df, threshold)
    drop_columns = set(df.columns) - set(columns)
    print(f"Dropping columns : {drop_columns}")
    return df.drop(drop_columns, axis=1)


def get_zero_var_cols(df):
    """
    Method to get all columns having zero variance
    """
    variance = df.var()
    # Check for numerical columns
    zero_var_columns = [variance[variance == 0].index.to_list()]
    # Check for Feature columns
    columns = df.select_dtypes(include='object').columns.to_list()
    for column in columns:
        if df[column].nunique == 1:
            zero_var_columns.append(column)

    return zero_var_columns


# def get_uniq_label_counts(df):
#     """
#     Get all unique labels count in the DataFrame
#     """
#     object_columns = get_dtype_columns(df)
#     return list(map(lambda x: (x, len(df[x].unique())), object_columns))
#
#
# def get_col_top_labels(series, top=10):
#     """
#     Get top n labels for the given Pandas Series
#     """
#     index = series.value_counts().head(top).index
#     return index.to_list()
#
#
# def top_one_hot_columns(df, top=10):
#     """
#     Returns a DataFrame of One Hot Encoded values of all Object columns
#     """
#     object_columns = get_dtype_columns(df)
#     one_hot_df = pd.DataFrame()
#     for col in object_columns:
#         for label in get_col_top_labels(df[col], top):
#             one_hot_col = str(col + '_' + label)
#             series = df[col]
#             if one_hot_col not in one_hot_df.columns.to_list():
#                 one_hot_df[one_hot_col] = np.where(series == label, 1, 0)
#
#     return one_hot_df


class CategoricalFeatures:
    def __init__(self, df, features, encoding_type='label', fillna_strategy='mode'):
        """
        Handle Categorical Features
        :param df: Pandas DataFrame
        :param features: Categorical Features
        :param encoding_type: encoding type
                label : Label encoding type
                binary : Binary encoding
                onehot : One Hot Encoding
                top-onehot : One Hot Encoding of Top 10 labels
        :param fillna_strategy: fill NA strategy explained below. Default "mode"
                mode -  mode of values
                new - create a new category unknown UKN for the null values
        """
        self.df = df
        self.features = features
        self.encoding_type = encoding_type
        self.fillna_strategy = fillna_strategy

        # Handle the NA
        self._handle_na()
        self.output_df = df.copy(deep=True)
        self.encoders = dict()

    def _handle_na(self):
        if self.fillna_strategy == 'new':
            for column in self.features:
                self.df[column] = self.df[column].astype(str).fillna('UKN')
        else:
            self.df = self.df[self.features].fillna('UKN')

    def _ordinal_encoding(self, columns):
        oe = preprocessing.OrdinalEncoder()
        for column in columns:
            self.encoders[column] = oe
            self.output_df[column] = oe.fit_transform(self.output_df[columns])
            self.output_df[column] = self.output_df[column].astype(int)

        return self.output_df

    def _label_encoding(self, columns):
        le = preprocessing.LabelEncoder()
        for column in columns:
            le.fit(self.df[column])
            self.encoders[column] = le
            self.output_df[column] = le.fit_transform(self.df[column])
            self.output_df[column] = self.output_df[column].astype(int)
        return self.output_df

    def _count_frequency_encoding(self, columns):
        """
        Perform top one-hot encoding for the labels
        """
        for column in columns:
            series = self.df[column]
            count_dict = series.value_counts().to_dict()
            self.encoders = count_dict
            self.output_df[column] = self.output_df[column].map(count_dict)
            self.encoders[column] = count_dict
        return self.output_df

    @staticmethod
    def get_col_top_labels(series, top=10):
        """
        Get top n labels for the given Pandas Series
        """
        index = series.value_counts().head(top).index
        return index.to_list()

    @staticmethod
    def get_uniq_labels_count(df, feature):
        """
        Get count of all the unique labels in a column
        :param df: Pandas DataFrame
        :param feature: Feature/Column
        :return: Cont of all unique labels
        """
        return len(df[feature].unique())

    @staticmethod
    def get_all_uniq_labels_count(df, features):
        """
        Get count of all the unique labels
        :param df: Pandas DataFrame
        :param features: Features/Columns
        :return: List of Tuples having columns and count of unique labels
        """
        return list(map(lambda x: (x, len(df[x].unique())), features))

    def _top_one_hot_encoding(self, columns):
        """
        Perform top one-hot encoding for the labels
        """
        for column in columns:
            series = self.df[column]
            top_labels = self.get_col_top_labels(series)
            for label in top_labels:
                one_hot_col = str(column + '_' + label)
                self.output_df[one_hot_col] = np.where(series == label, 1, 0)
            self.output_df.drop(column, axis=1, inplace=True)

        return self.output_df

    def _one_hot_encoding(self, columns):
        print('onehot encoding')
        # ohe = preprocessing.OneHotEncoder()
        for column in columns:
            onehot_df = pd.get_dummies(self.df[column], prefix=column)
            self.output_df = pd.concat([self.output_df, onehot_df])
            self.output_df.drop(column, axis=1, inplace=True)
        return self.output_df

    def _label_binarizer_encoding(self, columns):
        for column in columns:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[column].values)
            val = lbl.transform(self.df[column].values)
            self.output_df = self.output_df.drop(column, axis=1)
            for j in range(val.shape[1]):
                new_col_name = column + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            # self.binary_encoders[column] = lbl
        return self.output_df

    def fit_transform(self):
        if self.encoding_type == 'label':
            return self._label_encoding(self.features)
        elif self.encoding_type == 'ordinal':
            return self._ordinal_encoding(self.features)
        elif self.encoding_type == 'ohe':
            return self._one_hot_encoding(self.features)
        elif self.encoding_type == 'count_freq':
            return self._count_frequency_encoding(self.features)
        elif self.encoding_type == 'ohe-top':
            return self._top_one_hot_encoding(self.features)
        elif self.encoding_type == 'label_binarizer':
            return self._label_binarizer_encoding(self.features)
        else:
            raise Exception(f'{self.encoding_type} : Invalid value for encoding type')


class NumericalFeatures:
    def __init__(self, dataframe, encoding_features):
        """
        Handle Categorical Features
        :param dataframe: Pandas DataFrame
        :param encoding_features: Dictionary of {encoding_type:
        """
        self.df = dataframe
        self.encoding_features = encoding_features
        self.output_df = dataframe.copy(deep=True)
        self.fillna_strategy = self.get_fillna_strategy_list()
        self.encoders = self.get_encoders_list()
        # self.imputers = dict()
        self.scalers = dict()

    @staticmethod
    def get_encoders_list():
        enc_list = ['min-max', 'std', 'norm', 'robust', 'log', 'reciprocal', 'sqrt', 'exp', 'yeo-johnson',
                    'box-cox', 'quantile']
        return enc_list

    @staticmethod
    def get_fillna_strategy_list():
        strategy_list = ['mean', 'median', 'mode', 'most_frequent', 'end_distribution', 'knn', 'mice', 'softimpute']
        return strategy_list

    def _handle_na(self, columns, fillna_strategy):
        """
        Handle the missing values for Numerical Features
        :param columns: columns/features name in the dataframe
        :param fillna_strategy: NA handling strategy
        """
        if fillna_strategy in ['mean', 'median', 'most_frequent', 'mode']:
            # Change mode to most_frequent
            fillna_strategy = 'most_frequent' if fillna_strategy == 'mode' else fillna_strategy

            imp = SimpleImputer(missing_values=np.nan, strategy=fillna_strategy)
            self.output_df[columns] = imp.fit_transform(self.df[columns])
            # return self.imputers[column] = imp
        elif fillna_strategy == 'new':
            for column in columns:
                new_col_name = column + '_new'
                if self.output_df[column].isnull().count() > 0:
                    self.output_df[new_col_name] = np.where(self.output_df[column].isnull(), 1, 0)
        elif fillna_strategy == 'end_distribution':
            for column in columns:
                if self.output_df[column].isnull().count() > 0:
                    new_col_name = column + '_new'
                    extreme = self.df[column].mean() + 3 * self.df[column].std()
                    self.output_df[column] = self.output_df[column].fillna(extreme)
        elif fillna_strategy == 'mice':
            from fancyimpute import IterativeImputer
            imp = IterativeImputer()
            self.output_df[columns] = imp.fit_transform(self.output_df[columns])
            # self.imputers[columns] = imp
        elif fillna_strategy == 'knn':
            from fancyimpute import KNN
            imp = KNN()
            self.output_df[columns] = imp.fit_transform(self.output_df[columns])
            # self.imputers[column] = imp
        elif fillna_strategy == 'softimpute':
            from fancyimpute import SoftImpute
            imp = SoftImpute()
            self.output_df[columns] = imp.fit_transform(self.output_df[columns])
            # self.imputers[column] = imp

    def _std_scaling(self, columns):
        """
        Perform StandardScaler transformation
        """
        scaler = preprocessing.StandardScaler()
        self.scalers['std'] = scaler
        self.output_df[columns] = scaler.fit_transform(self.output_df[columns])

    def _minmax_scaling(self, columns):
        """
        Perform MinMax transformation
        """
        msc = preprocessing.MinMaxScaler()
        self.scalers['min-max'] = msc
        self.output_df[columns] = msc.fit_transform(self.output_df[columns])

    def _robust_scaling(self, columns):
        """
        Perform Robust transformation
        """
        scaler = preprocessing.RobustScaler()
        self.scalers['robust'] = scaler
        self.output_df[columns] = scaler.fit_transform(self.output_df[columns])

    def _box_cox_transform(self, columns):
        """
        Perform top Box-Cox transformation
        """
        transformer = preprocessing.PowerTransformer('box-cox')
        self.scalers['box-cox'] = transformer
        self.output_df[columns] = transformer.fit_transform(self.output_df[columns])

    def _yeo_johnson_transform(self, column):
        """
        Perform top Robust Scaling
        """
        transformer = preprocessing.PowerTransformer('yeo-johnson')
        self.scalers['robust'] = transformer
        self.output_df[column] = transformer.fit_transform(self.output_df[column])

    def _reciprocal_transform(self, columns):
        """ Perform Log transformation """
        for column in columns:
            self.output_df[column] = 1 / self.output_df[column]

    def _log_transform(self, columns):
        """ Perform Log transformation """
        for column in columns:
            self.output_df[column] = np.log(self.output_df[column])

    def _sqrt_transform(self, columns):
        """ Perform Square root transformation """
        for column in columns:
            self.output_df[column] = np.sqrt(self.output_df[column])

    def _exp_transform(self, columns):
        """ Perform Reciprocal transformation """
        for column in columns:
            self.output_df[column] = np.exp(self.output_df[column])

    def _quantile_transform(self, columns):
        """ Perform Quantile transformation """
        transformer = preprocessing.QuantileTransformer()
        self.scalers['quantile'] = transformer
        self.output_df[columns] = transformer.fit_transform(self.output_df[columns])

    def _normalize_transform(self, columns):
        """ Perform Quantile transformation """
        transformer = preprocessing.Normalizer()
        self.scalers['quantile'] = transformer
        self.output_df[columns] = transformer.transform(self.output_df[columns])

    def fit(self, ):
        for encoder, (columns, fillna_strategy) in self.encoding_features.items():
            # Check fillna_strategy and encoding types are valid
            if fillna_strategy not in self.fillna_strategy:
                raise Exception(f"Invalid fillna_strategy {fillna_strategy}. Valid values are {self.fillna_strategy}")

            if encoder not in self.encoders:
                raise Exception(f"Invalid Encoder {encoder}. Valid values are {self.encoders}")

        # Handle the missing values
        if fillna_strategy is not None:
            self._handle_na(columns, fillna_strategy)
        return self

    def fit_transform(self):
        # Call fit method
        self.fit()
        # Loop for each feature transformation
        for encoder, (columns, fillna_strategy) in self.encoding_features.items():
            if encoder == 'min-max':
                self._minmax_scaling(columns)
            elif encoder == 'std':
                self._std_scaling(columns)
            elif encoder == 'robust':
                self._robust_scaling(columns)
            elif encoder == 'log':
                self._log_transform(columns)
            elif encoder == 'norm':
                self._normalize_transform(columns)
            elif encoder == 'sqrt':
                self._sqrt_transform(columns)
            elif encoder == 'exp':
                self._exp_transform(columns)
            elif encoder == 'reciprocal':
                self._reciprocal_transform(columns)
            elif encoder == 'yeo-johnson':
                self._yeo_johnson_transform(columns)
            elif encoder == 'box-cox':
                self._box_cox_transform(columns)
            elif encoder == 'quantile':
                self._quantile_transform(columns)
            else:
                raise Exception(f'{encoder} : Invalid value for encoding type')

        return self.output_df


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    features = ['diagnosis']
    cat = CategoricalFeatures(df, features, 'ordinal', 'mode')
    transformed = cat.fit_transform()
    print(transformed[:6])

    encodings_list = [{'std-scale': (['diagnosis'], 'mean')}]
    num = NumericalFeatures(df, encodings_list)
