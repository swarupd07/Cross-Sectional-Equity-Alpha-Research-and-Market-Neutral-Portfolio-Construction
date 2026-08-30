import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_dataset(features: dict, forward_return: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({name: df.stack() for name, df in features.items()})
    y = forward_return.stack().rename("forward_return")
    joined = X.join(y, how="inner").dropna()
    return joined.drop(columns="forward_return"), joined["forward_return"]


class LinearAlphaModel:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(self.scaler.fit_transform(X), y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(self.scaler.transform(X)), index=X.index, name="pred")

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self.scaler.feature_names_in_)


class XGBoostAlphaModel:
    def __init__(
        self, n_estimators: int = 80, max_depth: int = 3, learning_rate: float = 0.05,
        subsample: float = 0.8, colsample_bytree: float = 0.8, random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index, name="pred")

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_)
