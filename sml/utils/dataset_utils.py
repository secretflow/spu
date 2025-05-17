# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split


from sml.utils.utils import get_logger

logger = get_logger(__name__)


# TODO: use more open source datasets by inlcuding ucimlrepo or other open source datasets
BI_CLASSIFICATION_OPEN_DATASETS = ["breast_cancer"]

MULTI_CLASSIFICATION_OPEN_DATASETS = ["irls"]

REGRESSION_OPEN_DATASETS = ["diabetes"]

# for clustering, we use mock dataset only now
CLUSTERING_OPEN_DATASETS = []

# for mtpl2, support 3 distributions: Poisson, Gamma, Tweedie
# for wine, support only Normal
# for bank_marketing, support only Bernoulli
# Note, currently, GLM not support Normal and Bernoulli
# GLM_NORMAL_DATASETS = ["wine"]
# GLM_BERNOULLI_DATASETS = ["bank_marketing"]
GLM_OPEN_DATASETS = ["mtpl2"]
# fetch datasset by mtpl2_{dist}, e.g. mtpl2_Poisson
_supported_glm_dist = ["Poisson", "Gamma", "Tweedie"]

# for metric datasets, we always use mock dataset
_metric_types = [
    # for classification rank metric, e.g. auc
    "bi_classification_rank",  # y_true is binary, y_pred is continuous
    # for classification metric, e.g. accuracy
    "bi_classification",  # both y_true and y_pred are binary
    # for multi-classification metric, e.g. accuracy
    "multi_classification",
    # for regression metric, e.g. mse
    "regression",
]
METRIC_OPEN_DATASETS = []


def preprocess_dataframe(
    df: pd.DataFrame,
    bin_feas: List[str] = [],
    bin_nums: List[int] = [],
    onehot_feas: List[str] = [],
    categ_drop_thres: int = 10,
    int_onehot_thres: int = 10,
    skip_cols: List[str] = ["_label"],
) -> pd.DataFrame:
    """A function to pre-process the dataframe.

    Args:
        df (pd.DataFrame): dataset
        bin_feas (list, optional): some features need to be binned. Defaults to [].
        bin_nums (list, optional): binned numbers for each feature. Defaults to [].
        onehot_feas (list, optional): some features need to be one-hot encoded. Defaults to [].
        categ_drop_thres (int, optional): threshold for dropping categorical columns. Defaults to 10.
        int_onehot_thres (int, optional): threshold for one-hot encoding integer columns. Defaults to 10.
        skip_cols (list, optional): columns to skip. Defaults to ["_label"].
    )

    Returns:
        pd.DataFrame:
    """
    assert len(bin_feas) == len(bin_nums)
    nrows = df.shape[0]

    drop_lsts = []
    # step 1: delete rows with missing values < 0.1%
    for column in df.columns:
        nan_count = df[column].isna().sum()
        if nan_count == 0:
            continue
        if nan_count < nrows * 0.001:
            drop_lsts.append(column)
    df = df.dropna(subset=drop_lsts)

    # step 2: if numbers of missing values >= 30%, then add a new column
    # step 3: else, use mode to fill missing values
    for column in df.columns:
        nan_count = df[column].isna().sum()
        if nan_count == 0:
            continue
        if nan_count >= nrows * 0.3:
            df[column + '_is_nan'] = df[column].isna().astype(int)
        elif nan_count > 0:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)

    # step 4: binning or one-hot encoding for each feature
    for column in df.columns:
        un = df[column].nunique()
        # if unique values = 1, drop it
        if un == 1:
            df = df.drop(columns=[column])

        # check if column is skipped
        if column in skip_cols:
            assert df[column].dtype != "O"  # O means object (str)
            continue

        #  check if column is onehot
        if column in onehot_feas:
            df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
            continue
        # check if column is binned
        if column in bin_feas:
            assert df[column].dtype != "O"
            ix = bin_feas.index(column)
            df[column] = pd.cut(df[column], bins=bin_nums[ix], labels=False)
            continue

        if df[column].dtype == 'O':
            if un > categ_drop_thres:  # if too many categories, drop directly
                df = df.drop(columns=[column])
            else:
                df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
        elif df[column].dtype == 'float':
            pass  # leave float columns
        elif df[column].dtype == 'int':
            # if unique values = 2, map to 0,1
            if un == 2:
                uni = np.sort(df[column].unique())
                df[column] = df[column].map(lambda x: 1 if x == uni[1] else 0)
            # if unique values <= 5, one-hot encode
            elif un <= int_onehot_thres:
                df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
            else:
                # else, leave it for following encoding
                pass

    # step5: do min-max normalization
    for column in df.columns:
        if column in skip_cols:
            continue
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)

    return df


class DataRecord:
    def __init__(self, values, name, meta: "DataMeta"):
        self.name = name
        self.values = values
        self.meta = meta

    def __str__(self):
        return f"DataRecord(name={self.name}, " f"meta={self.meta})"

    def __repr__(self):
        return self.__str__()


# TODO: add more metas
class DataMeta:
    def __init__(self, is_mock, train_shape, test_shape=None, others=None):
        self.is_mock = is_mock
        self.train_shape = train_shape
        # only valid in traindional-ML tasks
        self.test_shape = test_shape

        if others is None:
            others = {}
        assert isinstance(others, dict), "others should be a dict"
        self.others = others

    def __str__(self):
        return (
            f"DataMeta(is_mock={self.is_mock}, "
            f"train_shape={self.train_shape}, "
            f"test_shape={self.test_shape}), "
            f"others={self.others}"
        )

    def __repr__(self):
        return self.__str__()


def make_data_record(values, name, meta) -> DataRecord:
    return DataRecord(values, name, meta)


def make_data_meta(
    is_mock: bool,
    train_shape: tuple,
    test_shape: Optional[tuple] = None,
    others: Optional[dict] = None,
) -> DataMeta:
    return DataMeta(is_mock, train_shape, test_shape=test_shape, others=others)


class DataSetsFactory:
    def __init__(
        self,
        use_open: bool = True,
        test_size: float = 0.2,
        use_mock: bool = False,
        mock_sample_sizes: List[int] = [10_000, 100_000],
        mock_feature_sizes: List[int] = [10, 20],
        mock_multi_classes: List[int] = [3, 4],
        mock_clustring_centers: List[int] = [3, 6],
        mock_decomposition_factors: List[int] = [2, 4],
        mock_decomposition_sample_sizes: List[int] = [100, 1000],
        random_state: int = 107,
    ) -> None:
        assert (
            use_mock or use_open
        ) == True, "use_mock and use_open cannot be both False"

        assert len(mock_sample_sizes) == len(
            mock_feature_sizes
        ), "mock_sample_sizes and mock_feature_sizes should have the same length"

        assert len(mock_sample_sizes) == len(mock_multi_classes)
        assert len(mock_sample_sizes) == len(mock_clustring_centers)
        assert len(mock_decomposition_factors) == len(mock_decomposition_sample_sizes)
        # TODO: use large sample size for decomposition
        assert (
            mock_decomposition_sample_sizes[-1] <= 5000
        ), "decomposition sample size should be small numbers now."

        # Note: name starting with "mock_" are mock datasets for efficiency test
        self.use_mock = use_mock
        self.mock_sample_sizes = mock_sample_sizes
        self.mock_feature_sizes = mock_feature_sizes
        self.mock_nb = len(mock_sample_sizes)
        self.mock_multi_classes = mock_multi_classes
        self.mock_clustring_centers = mock_clustring_centers
        self.mock_decomposition_factors = mock_decomposition_factors
        self.mock_decomposition_sample_sizes = mock_decomposition_sample_sizes

        # use open source dataset for accuracy test
        self.use_open = use_open
        self.test_size = test_size

        # for re-producibility
        self.random_state = random_state

        # 1. normal ML-aspect datasets
        # for each dataset, return [x_train, x_test, y_train, y_test]
        self.bi_classification_datasets = {}
        self.multi_classification_datasets = {}
        self.regression_datasets = {}

        # 2. some special task datasets
        # for clustering, return [x, y]
        self.clustering_datasets = {}

        # TODO: add weights for some metrics
        # for metric computation, return [y_true, y_pred]
        # Just a functional class/function, we always use mock dataset
        self.metric_datasets = {}

        # for GLM datasets, return [x_train, x_test, y_train, y_test, w_train, w_test]
        # mock dataset is exactly open source mptl2 dataset
        self.glm_datasets = {}

        # for decompostion datasets, return [x]
        # we only check the explained variance and simple orthogonal property
        self.decomposition_datasets = {}

        # TODO: not collected yet
        self.multi_label_datasets = {}

    @classmethod
    def gen_mock_sample_weights(cls, n_samples):
        un_weighted = np.random.rand(n_samples) + 1e-6
        return un_weighted / np.sum(un_weighted)

    def load_bi_classification_datasets(self):
        if self.bi_classification_datasets:
            return self.bi_classification_datasets

        if self.use_open:
            for data_name in BI_CLASSIFICATION_OPEN_DATASETS:
                values = load_open_source_datasets(
                    name=data_name,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(
                    is_mock=False,
                    train_shape=values[0].shape,
                    test_shape=values[1].shape,
                )
                self.bi_classification_datasets[data_name] = make_data_record(
                    values=values, name=data_name, meta=meta
                )
                logger.info(f"load open source data: [{data_name}] success.")

        if self.use_mock:
            for i in range(self.mock_nb):
                mock_name = f"mock_bi_classification_dataset_{i}"
                values = mock_bi_classification(
                    n_samples=self.mock_sample_sizes[i],
                    n_features=self.mock_feature_sizes[i],
                    random_seed=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
                self.bi_classification_datasets[mock_name] = make_data_record(
                    values=values, name=mock_name, meta=meta
                )
                logger.info(f"load mock dataset: [{mock_name}] success")

        return self.bi_classification_datasets

    def load_multi_classification_datasets(self):
        if self.multi_classification_datasets:
            return self.multi_classification_datasets

        if self.use_open:
            for data_name in MULTI_CLASSIFICATION_OPEN_DATASETS:
                values = load_open_source_datasets(
                    name=data_name,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(
                    is_mock=False,
                    train_shape=values[0].shape,
                    test_shape=values[1].shape,
                )
                self.multi_classification_datasets[data_name] = make_data_record(
                    values=values, name=data_name, meta=meta
                )

                logger.info(f"load open source data: [{data_name}] success.")

        if self.use_mock:
            for i in range(self.mock_nb):
                mock_name = f"mock_multi_classification_dataset_{i}"
                values = mock_multi_classification(
                    n_samples=self.mock_sample_sizes[i],
                    n_features=self.mock_feature_sizes[i],
                    n_classes=self.mock_multi_classes[i],
                    random_seed=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
                self.multi_classification_datasets[mock_name] = make_data_record(
                    values=values, name=mock_name, meta=meta
                )
                logger.info(f"load mock dataset: [{mock_name}] success")

        return self.multi_classification_datasets

    def load_regression_datasets(self):
        if self.regression_datasets:
            return self.regression_datasets

        if self.use_open:
            for data_name in REGRESSION_OPEN_DATASETS:
                values = load_open_source_datasets(
                    name=data_name,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(
                    is_mock=False,
                    train_shape=values[0].shape,
                    test_shape=values[1].shape,
                )
                self.regression_datasets[data_name] = make_data_record(
                    values=values, name=data_name, meta=meta
                )
                logger.info(f"load open source data: [{data_name}] success.")

        if self.use_mock:
            for i in range(self.mock_nb):
                mock_name = f"mock_regression_dataset_{i}"
                values = mock_regression(
                    n_samples=self.mock_sample_sizes[i],
                    n_features=self.mock_feature_sizes[i],
                    random_seed=self.random_state,
                    need_split_train_test=True,
                )
                meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
                self.regression_datasets[mock_name] = make_data_record(
                    values=values, name=mock_name, meta=meta
                )
                logger.info(f"load mock dataset: [{mock_name}] success")

        return self.regression_datasets

    def load_clustering_datasets(self):
        if self.clustering_datasets:
            return self.clustering_datasets

        for i in range(self.mock_nb):
            mock_name = f"mock_clustering_dataset_{i}"
            values = mock_clustering(
                n_samples=self.mock_sample_sizes[i],
                n_features=self.mock_feature_sizes[i],
                centers=self.mock_clustring_centers[i],
                random_seed=self.random_state,
            )
            meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
            self.clustering_datasets[mock_name] = make_data_record(
                values=values, name=mock_name, meta=meta
            )
            logger.info(f"load mock dataset: [{mock_name}] success")

        return self.clustering_datasets

    def load_metric_datasets(self):
        if self.metric_datasets:
            return self.metric_datasets

        # for metric, always use mock dataset
        for metric_type in _metric_types:
            mock_name = f"mock_metric_{metric_type}"
            values = mock_metrics(
                metric_type=metric_type,
                n_samples=self.mock_sample_sizes[-1],  # always use the last size
                random_seed=self.random_state,
                multi_classes=self.mock_multi_classes[-1],  # always use the last size
            )
            meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
            self.metric_datasets[mock_name] = make_data_record(
                values=values, name=mock_name, meta=meta
            )
            logger.info(f"load mock dataset: [{mock_name}] success")

        # for glm metric
        for dist in _supported_glm_dist:
            mock_name = f"mock_metric_glm_{dist}"
            values = mock_metrics(
                metric_type="glm",
                n_samples=self.mock_sample_sizes[-1],  # always use the last size
                random_seed=self.random_state,
                distribution=dist,
            )
            meta = make_data_meta(is_mock=True, train_shape=values[0].shape)
            self.metric_datasets[mock_name] = make_data_record(
                values=values, name=mock_name, meta=meta
            )
            logger.info(f"load mock dataset: [{mock_name}] success")

        return self.metric_datasets

    def load_glm_datasets(self):
        if self.glm_datasets:
            return self.glm_datasets

        # for glm, we only use open source dataset
        for dist in _supported_glm_dist:
            data_name = f"mtpl2_{dist}"
            values = load_open_source_datasets(
                name=data_name,
                test_size=self.test_size,
                random_state=self.random_state,
                need_split_train_test=True,
            )
            meta = make_data_meta(
                is_mock=False, train_shape=values[0].shape, test_shape=values[1].shape
            )
            self.glm_datasets[data_name] = make_data_record(
                values=values, name=data_name, meta=meta
            )
            logger.info(f"load open source data: [{data_name}] success.")

        return self.glm_datasets

    def load_decomposition_datasets(self):
        if self.decomposition_datasets:
            return self.decomposition_datasets

        mock_nb = len(self.mock_decomposition_factors)
        for i in range(mock_nb):
            mock_name = f"mock_decomposition_dataset_{i}"
            values = mock_decomposition(
                n_samples=self.mock_decomposition_sample_sizes[i],
                n_factors=self.mock_decomposition_factors[i],
                random_seed=self.random_state,
            )
            meta = make_data_meta(
                is_mock=True,
                train_shape=values.shape,
                others={"n_factors": self.mock_decomposition_factors[i]},
            )
            self.decomposition_datasets[mock_name] = make_data_record(
                values=values, name=mock_name, meta=meta
            )
            logger.info(f"load mock dataset: [{mock_name}] success")

        return self.decomposition_datasets

    def load_multi_label_datasets(self):
        raise NotImplementedError("not implemented yet.")


def load_open_source_datasets(
    name, test_size=0.2, random_state=107, need_split_train_test=True, **kwargs
):
    """Load pre-defined open source datasets.

    This function loads pre-processed versions of popular open source datasets
    for machine learning tasks. The datasets are processed and can be optionally
    split into training and testing sets.

    Args:
        name (str): Name of the dataset to load. Supported datasets:
            - "breast_cancer": Binary classification dataset from sklearn
            - "irls": Multi-class classification dataset (Iris) from sklearn
            - "diabetes": Regression dataset from sklearn
            - "mtpl2_Poisson", "mtpl2_Gamma", "mtpl2_Tweedie": GLM datasets with
              different distribution assumptions
        test_size (float, optional): Proportion of the dataset to include in the test split.
            Only used when need_split_train_test=True. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 107.
        need_split_train_test (bool, optional): Whether to split data into train and test
            sets. Defaults to True.
        **kwargs: Additional arguments to pass to the dataset loading function.

    Returns:
        Various formats depending on the dataset and need_split_train_test parameter:

        If need_split_train_test=True:
            For standard ML datasets (classification/regression):
                (x_train, x_test, y_train, y_test) where:
                - x_train: Training features, numpy array
                - x_test: Test features, numpy array
                - y_train: Training labels/targets, numpy array
                - y_test: Test labels/targets, numpy array

            For GLM datasets (mtpl2_*):
                (x_train, x_test, y_train, y_test, w_train, w_test) where:
                - x_train, x_test: Training/test features
                - y_train, y_test: Training/test targets
                - w_train, w_test: Training/test sample weights

        If need_split_train_test=False:
            For standard ML datasets:
                (X, y) where:
                - X: Features, numpy array
                - y: Labels/targets, numpy array

            For GLM datasets:
                (X, y, w) where:
                - X: Features, numpy array
                - y: Targets, numpy array
                - w: Sample weights, numpy array

    Raises:
        ValueError: If the requested dataset is not found.

    Examples:
        >>> # Load breast cancer dataset with train/test split
        >>> x_train, x_test, y_train, y_test = load_open_source_datasets(
        ...     name="breast_cancer",
        ...     test_size=0.2,
        ...     random_state=42
        ... )
        >>>
        >>> # Load GLM dataset with Poisson distribution
        >>> x_train, x_test, y_train, y_test, w_train, w_test = load_open_source_datasets(
        ...     name="mtpl2_Poisson"
        ... )
    """
    func_name = f"fetch_and_preprocess_{name}"
    func = globals().get(func_name)

    if callable(func):
        return func(
            test_size=test_size,
            random_state=random_state,
            need_split_train_test=need_split_train_test,
            **kwargs,
        )

    raise ValueError(f"function {func_name} not found.")


###################################
### 1. normal ML-aspect datasets
###################################


# regression dataset
def fetch_and_preprocess_diabetes(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    ds = sk_datasets.load_diabetes()
    x, y = ds["data"], ds["target"]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    if need_split_train_test:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test
    else:
        return x, y


# multi-class/clustering classification dataset
def fetch_and_preprocess_irls(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    ds = sk_datasets.load_iris()
    x, y = ds["data"], ds["target"]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    if need_split_train_test:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test
    else:
        return x, y


# binary classification dataset
def fetch_and_preprocess_breast_cancer(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    ds = sk_datasets.load_breast_cancer()
    x, y = ds['data'], ds['target']
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    if need_split_train_test:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test
    else:
        return x, y


###################################
### 2. special datasets
###################################


# ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims
def _load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = sk_datasets.fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = sk_datasets.fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


def _preprocess_mtpl2(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        KBinsDiscretizer,
        OneHotEncoder,
        StandardScaler,
    )

    # Correct for unreasonable observations (that might be data error)
    # and a few exceptionally large claim amounts
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)
    # If the claim amount is 0, then we do not count it as a claim. The loss function
    # used by the severity model needs strictly positive claim amounts. This way
    # frequency and severity are more consistent with each other.
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )

    column_trans = ColumnTransformer(
        [
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, subsample=int(2e5), random_state=0),
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    X = column_trans.fit_transform(df)

    # Insurances companies are interested in modeling the Pure Premium, that is
    # the expected total claim amount per unit of exposure for each policyholder
    # in their portfolio:
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

    # This can be indirectly approximated by a 2-step modeling: the product of the
    # Frequency times the average claim amount per claim:
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

    return X, df


# glm datasets
def _fetch_and_preprocess_mtpl2(
    test_size=0.2, random_state=107, distribution="Tweedie", need_split_train_test=True
):
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df = _load_mtpl2()

    X, df = _preprocess_mtpl2(df)

    if need_split_train_test:
        df_train, df_test, X_train, X_test = train_test_split(
            df, X, random_state=random_state, test_size=test_size
        )

        if distribution == "Poisson":
            # The number of claims (``ClaimNb``) is a positive integer (0 included).
            # Thus, this target can be modelled by a Poisson distribution.
            # It is then assumed to be the number of discrete events occurring with a
            # constant rate in a given time interval (``Exposure``, in units of years).
            # Here we model the frequency ``y = ClaimNb / Exposure``, which is still a
            # (scaled) Poisson distribution, and use ``Exposure`` as `sample_weight`.
            return (
                X_train.toarray(),
                X_test.toarray(),
                df_train["Frequency"].to_numpy(),
                df_test["Frequency"].to_numpy(),
                df_train["Exposure"].to_numpy(),
                df_test["Exposure"].to_numpy(),
            )
        elif distribution == "Gamma":
            # The mean claim amount or severity (`AvgClaimAmount`) can be empirically
            # shown to follow approximately a Gamma distribution. We fit a GLM model for
            # the severity with the same features as the frequency model.
            #
            # Note:
            #
            # - We filter out ``ClaimAmount == 0`` as the Gamma distribution has support
            #   on :math:`(0, \infty)`, not :math:`[0, \infty)`.
            # - We use ``ClaimNb`` as `sample_weight` to account for policies that contain
            #   more than one claim.
            mask_train = df_train["ClaimAmount"] > 0
            mask_test = df_test["ClaimAmount"] > 0
            return (
                X_train[mask_train.values].toarray(),
                X_test[mask_test.values].toarray(),
                df_train.loc[mask_train, "AvgClaimAmount"].to_numpy(),
                df_test.loc[mask_test, "AvgClaimAmount"].to_numpy(),
                df_train.loc[mask_train, "ClaimNb"].to_numpy(),
                df_test.loc[mask_test, "ClaimNb"].to_numpy(),
            )
        elif distribution == "Tweedie":
            return (
                X_train.toarray(),
                X_test.toarray(),
                df_train["PurePremium"].to_numpy(),
                df_test["PurePremium"].to_numpy(),
                df_train["Exposure"].to_numpy(),
                df_test["Exposure"].to_numpy(),
            )
        else:
            raise ValueError(f"Not supported for {distribution} yet!")
    else:
        if distribution == "Poisson":
            return (
                X.toarray(),
                df["Frequency"].to_numpy(),
                df["Exposure"].to_numpy(),
            )
        elif distribution == "Gamma":
            mask = df["ClaimAmount"] > 0
            return (
                X[mask.values].toarray(),
                df.loc[mask, "AvgClaimAmount"].to_numpy(),
                df.loc[mask, "ClaimNb"].to_numpy(),
            )
        elif distribution == "Tweedie":
            return (
                X.toarray(),
                df["PurePremium"].to_numpy(),
                df["Exposure"].to_numpy(),
            )
        else:
            raise ValueError(f"Not supported for {distribution} yet!")


def fetch_and_preprocess_mtpl2_Poisson(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    return _fetch_and_preprocess_mtpl2(
        test_size=test_size,
        random_state=random_state,
        distribution="Poisson",
        need_split_train_test=need_split_train_test,
    )


def fetch_and_preprocess_mtpl2_Gamma(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    return _fetch_and_preprocess_mtpl2(
        test_size=test_size,
        random_state=random_state,
        distribution="Gamma",
        need_split_train_test=need_split_train_test,
    )


def fetch_and_preprocess_mtpl2_Tweedie(
    test_size=0.2, random_state=107, need_split_train_test=True
):
    return _fetch_and_preprocess_mtpl2(
        test_size=test_size,
        random_state=random_state,
        distribution="Tweedie",
        need_split_train_test=need_split_train_test,
    )


###################################
### 3. mock datasets
###################################


def load_mock_datasets(
    n_samples: int,
    task_type: str,
    random_seed: int = 107,
    **kwargs,
):
    """Load mock datasets based on different task types.

    This function generates synthetic datasets for various machine learning tasks using
    scikit-learn's data generation utilities. The datasets can be used for testing and
    benchmarking machine learning algorithms.

    Args:
        n_samples (int): Number of samples to generate.
        task_type (str): Type of task. Supported types:
            - "bi_classification": Binary classification task
            - "multi_classification": Multi-class classification task
            - "regression": Regression task
            - "clustering": Clustering task
            - "metric": Metric computation task
            - "glm": Generalized Linear Model task (not supported for mock data)
            - "decomposition": Decomposition task
        random_seed (int, optional): Random seed for reproducibility. Defaults to 107.
        **kwargs: Additional arguments depending on the task_type:
            - For "bi_classification":
                n_features (int): Number of features.
                test_size (float, optional): Test set proportion. Defaults to 0.2.
                need_split_train_test (bool, optional): Whether to split data into
                    train and test sets. Defaults to True.

                Returns:
                    If need_split_train_test=True:
                        (x_train, x_test, y_train, y_test) where:
                        - x_train: Training features, numpy array of shape (n_samples*(1-test_size), n_features)
                        - x_test: Test features, numpy array of shape (n_samples*test_size, n_features)
                        - y_train: Training labels, numpy array of shape (n_samples*(1-test_size),) with binary values (0 or 1)
                        - y_test: Test labels, numpy array of shape (n_samples*test_size,) with binary values (0 or 1)
                    If need_split_train_test=False:
                        (X, y) where:
                        - X: Features, numpy array of shape (n_samples, n_features)
                        - y: Labels, numpy array of shape (n_samples,) with binary values (0 or 1)

            - For "multi_classification":
                n_features (int): Number of features.
                n_classes (int, optional): Number of classes. Defaults to 3.
                n_informative (int, optional): Number of informative features. This should be high enough
                    to satisfy the constraint: n_classes * n_clusters_per_class <= 2^n_informative.
                    For multi-class scenarios, it's recommended to set this value higher than the default.
                    Defaults to 2.
                test_size (float, optional): Test set proportion. Defaults to 0.2.
                need_split_train_test (bool, optional): Whether to split data into
                    train and test sets. Defaults to True.

                Returns:
                    If need_split_train_test=True:
                        (x_train, x_test, y_train, y_test) where:
                        - x_train: Training features, numpy array of shape (n_samples*(1-test_size), n_features)
                        - x_test: Test features, numpy array of shape (n_samples*test_size, n_features)
                        - y_train: Training labels, numpy array of shape (n_samples*(1-test_size),) with values from 0 to n_classes-1
                        - y_test: Test labels, numpy array of shape (n_samples*test_size,) with values from 0 to n_classes-1
                    If need_split_train_test=False:
                        (X, y) where:
                        - X: Features, numpy array of shape (n_samples, n_features)
                        - y: Labels, numpy array of shape (n_samples,) with values from 0 to n_classes-1

            - For "regression":
                n_features (int): Number of features.
                test_size (float, optional): Test set proportion. Defaults to 0.2.
                need_split_train_test (bool, optional): Whether to split data into
                    train and test sets. Defaults to True.

                Returns:
                    If need_split_train_test=True:
                        (x_train, x_test, y_train, y_test) where:
                        - x_train: Training features, numpy array of shape (n_samples*(1-test_size), n_features)
                        - x_test: Test features, numpy array of shape (n_samples*test_size, n_features)
                        - y_train: Training target values, numpy array of shape (n_samples*(1-test_size),) with continuous values
                        - y_test: Test target values, numpy array of shape (n_samples*test_size,) with continuous values
                    If need_split_train_test=False:
                        (X, y) where:
                        - X: Features, numpy array of shape (n_samples, n_features)
                        - y: Target values, numpy array of shape (n_samples,) with continuous values

            - For "clustering":
                n_features (int): Number of features.
                centers (int): Number of cluster centers.

                Returns:
                    (X, y) where:
                    - X: Features, numpy array of shape (n_samples, n_features)
                    - y: True cluster labels, numpy array of shape (n_samples,) with values from 0 to centers-1

            - For "metric":
                metric_type (str): Type of metric, supported types:
                    - "bi_classification_rank": Binary classification ranking metric (e.g., AUC)
                    - "bi_classification": Binary classification metric (e.g., accuracy)
                    - "multi_classification": Multi-class classification metric
                    - "regression": Regression metric (e.g., MSE)
                    - "glm": GLM metric
                multi_classes (int, optional): Number of classes for multi-class metric. Defaults to 5.
                distribution (str, optional): Distribution type for GLM metric, must be one of
                    ["Poisson", "Gamma", "Tweedie"].

                Returns:
                    (y_true, y_pred) where:
                    - y_true: True labels/values, numpy array of shape (n_samples,)
                    - y_pred: Predicted labels/scores, numpy array of shape (n_samples,)

            - For "decomposition":
                n_factors (int): Number of factors, used to generate a data matrix with specific characteristics.

                Returns:
                    Y: Numpy array of shape (n_samples, 10*n_factors) whose covariance matrix
                       has n_factors large eigenvalues.

    Returns:
        Various formats depending on the task_type and need_split_train_test parameter.
        See detailed return values for each task type above.

    Examples:
        >>> # Generate binary classification data with train/test split
        >>> x_train, x_test, y_train, y_test = load_mock_datasets(
        ...     n_samples=1000,
        ...     task_type="bi_classification",
        ...     n_features=20
        ... )
        >>>
        >>> # Generate binary classification data without train/test split
        >>> X, y = load_mock_datasets(
        ...     n_samples=1000,
        ...     task_type="bi_classification",
        ...     n_features=20,
        ...     need_split_train_test=False
        ... )
        >>>
        >>> # Generate multi-class data
        >>> x_train, x_test, y_train, y_test = load_mock_datasets(
        ...     n_samples=1000,
        ...     task_type="multi_classification",
        ...     n_features=20,
        ...     n_classes=5
        ... )
        >>>
        >>> # Generate clustering data
        >>> X, y = load_mock_datasets(
        ...     n_samples=1000,
        ...     task_type="clustering",
        ...     n_features=20,
        ...     centers=5
        ... )
    """
    if task_type == "bi_classification":
        n_features = kwargs.get("n_features", None)
        assert n_features is not None, "n_features is required for bi_classification"
        test_size = kwargs.get("test_size", 0.2)
        need_split_train_test = kwargs.get("need_split_train_test", True)
        return mock_bi_classification(
            n_samples, n_features, random_seed, test_size, need_split_train_test
        )
    elif task_type == "multi_classification":
        n_features = kwargs.get("n_features", None)
        assert n_features is not None, "n_features is required for multi_classification"
        n_classes = kwargs.get("n_classes", 3)
        n_informative = kwargs.get("n_informative", 2)
        test_size = kwargs.get("test_size", 0.2)
        need_split_train_test = kwargs.get("need_split_train_test", True)
        return mock_multi_classification(
            n_samples,
            n_features,
            n_classes,
            n_informative,
            random_seed,
            test_size,
            need_split_train_test,
        )
    elif task_type == "regression":
        n_features = kwargs.get("n_features", None)
        assert n_features is not None, "n_features is required for regression"
        test_size = kwargs.get("test_size", 0.2)
        need_split_train_test = kwargs.get("need_split_train_test", True)
        return mock_regression(
            n_samples, n_features, random_seed, test_size, need_split_train_test
        )
    elif task_type == "clustering":
        n_features = kwargs.get("n_features", None)
        assert n_features is not None, "n_features is required for clustering"
        centers = kwargs.get("centers", None)
        assert centers is not None, "centers is required for clustering"
        return mock_clustering(n_samples, n_features, centers, random_seed)
    elif task_type == "metric":
        metric_type = kwargs.get("metric_type", None)
        assert metric_type is not None, "metric_type is required for metric"
        kwargs.pop("metric_type")
        return mock_metrics(metric_type, n_samples, random_seed, **kwargs)
    elif task_type == "glm":
        raise NotImplementedError("Not supported for mock glm yet!")
    elif task_type == "decomposition":
        n_factors = kwargs.get("n_factors", None)
        assert n_factors is not None, "n_factors is required for decomposition"
        return mock_decomposition(n_samples, n_factors, random_seed)
    else:
        raise ValueError(f"Not supported for {task_type} yet!")


def mock_bi_classification(
    n_samples, n_features, random_seed=107, test_size=0.2, need_split_train_test=True
):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples,
        n_features,
        n_classes=2,
        random_state=random_seed,
    )

    if need_split_train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_seed)
    else:
        return X, y


def mock_multi_classification(
    n_samples,
    n_features,
    n_classes=3,
    n_informative=2,
    random_seed=107,
    test_size=0.2,
    need_split_train_test=True,
):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples,
        n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=random_seed,
    )

    if need_split_train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_seed)
    else:
        return X, y


def mock_regression(
    n_samples, n_features, random_seed=107, test_size=0.2, need_split_train_test=True
):
    from sklearn.datasets import make_regression

    # we leave the coef for future use
    X, y, _ = make_regression(
        n_samples, n_features, random_state=random_seed, coef=True
    )

    if need_split_train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_seed)
    else:
        return X, y


def mock_clustering(n_samples, n_features, centers, random_seed=107):
    from sklearn.datasets import make_blobs

    # we leave the coef for future use
    X, y, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_seed,
        return_centers=True,
    )
    return X, y


def mock_metrics(metric_type, n_samples, random_seed=107, **kwargs):
    rnd = np.random.RandomState(random_seed)
    if metric_type == "bi_classification_rank":
        y_true = rnd.choice([0, 1], size=n_samples)
        y_pred = rnd.rand(n_samples)
    elif metric_type == "bi_classification":
        y_true = rnd.choice([0, 1], size=n_samples)
        y_pred = rnd.choice([0, 1], size=n_samples)
    elif metric_type == "multi_classification":
        multi_classes = int(kwargs.get("multi_classes", 5))
        y_true = rnd.choice(range(multi_classes), size=n_samples)
        y_pred = rnd.choice(range(multi_classes), size=n_samples)
    elif metric_type == "regression":
        y_true = rnd.randn(n_samples)
        y_pred = rnd.randn(n_samples)
    elif metric_type == "glm":
        distribution = kwargs.get("distribution")
        assert distribution in _supported_glm_dist
        if distribution == "Poisson":
            y_true = rnd.poisson(lam=1, size=n_samples)
            y_pred = rnd.poisson(lam=1, size=n_samples)
        elif distribution == "Gamma":
            y_true = rnd.gamma(shape=1, scale=1, size=n_samples)
            y_pred = rnd.gamma(shape=1, scale=1, size=n_samples)
        elif distribution == "Tweedie":
            # TODO: generate a generic Tweedie distribution is hard
            y_true = rnd.rand(n_samples)
            y_pred = rnd.rand(n_samples)
        else:
            raise NotImplementedError(f"Not supported for {distribution} yet!")

        # for tweedie deviance, need positive
        y_true = np.abs(y_true) + 1e-6
        y_pred = np.abs(y_pred) + 1e-6
    elif metric_type == "clustering":
        raise NotImplementedError(f"Not supported for {metric_type} yet!")
    else:
        raise ValueError(f"Not supported for {metric_type}.")

    return y_true, y_pred


# glm only uses open source datasets
def mock_glm():
    raise NotImplementedError("Not supported for mock glm yet!")


def mock_decomposition(n_samples, n_factors, random_seed=107) -> np.ndarray:
    rnd = np.random.RandomState(random_seed)

    # TODO: now, we fix the total features to 10*n_factors
    total_feas = 10 * n_factors
    max_eignvals_lo, min_eignvals_hi = 50.0, 150.0
    small_eignvals_mi = 1.0

    large_eignvals = rnd.uniform(
        low=max_eignvals_lo, high=min_eignvals_hi, size=n_factors
    )
    small_eignvals = rnd.uniform(
        low=small_eignvals_mi / 2, high=max_eignvals_lo, size=total_feas - n_factors
    )

    # 1. define the diagonal eignvals matrix
    eignvals = np.diag(np.concatenate([large_eignvals, small_eignvals]))

    # 2. generate a random positive matrix
    Q, _ = np.linalg.qr(rnd.randn(total_feas, total_feas))

    # 3. generate the matrix Y that its cov matrix has approximately the same eignvals
    X = rnd.randn(n_samples, total_feas)
    Y = X @ Q @ np.sqrt(eignvals)

    return Y
