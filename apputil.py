import numpy as np
import pandas as pd

class GroupEstimate:
    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self._group_map = None
        self._cols = None
        self._default_category = None
        self._default_map = None

    def fit(self, X, y, default_category=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = pd.Series(y).reset_index(drop=True)

        if len(X_df) != len(y_ser):
            raise ValueError("X and y must have the same length")

        self._cols = list(X_df.columns)
        self._default_category = default_category

        combined = X_df.copy().reset_index(drop=True)
        combined["_y"] = y_ser.values

        # Use the string name for aggregation to avoid pandas FutureWarning
        agg_name = "mean" if self.estimate == "mean" else "median"
        grouped = combined.groupby(self._cols, observed=True)
        group_stats = grouped["_y"].agg(agg_name)

        group_map = {}
        for key, val in group_stats.items():
            if not isinstance(key, tuple):
                key = (key,)
            group_map[tuple(key)] = float(val)
        self._group_map = group_map

        if default_category is not None:
            if default_category not in self._cols:
                raise ValueError("default_category must be one of the columns in X")
            default_grouped = combined.groupby(default_category, observed=True)["_y"].agg(agg_name)
            default_map = {k: float(v) for k, v in default_grouped.items()}
            self._default_map = default_map
        else:
            self._default_map = None

        return self

    def predict(self, X_):
        if self._group_map is None or self._cols is None:
            raise ValueError("Model has not been fitted. Call fit(X, y) before predict().")

        if isinstance(X_, pd.DataFrame):
            Xp = X_.copy().reset_index(drop=True)
            if list(Xp.columns) != self._cols:
                if set(Xp.columns) == set(self._cols):
                    Xp = Xp[self._cols]
                else:
                    raise ValueError("Predict input columns must match the columns used in fit")
        else:
            Xp = pd.DataFrame(X_, columns=self._cols)

        results = []
        missing_count = 0

        for _, row in Xp.iterrows():
            key = tuple(row[col] for col in self._cols)
            if key in self._group_map:
                results.append(self._group_map[key])
                continue

            used_default = False
            if self._default_map is not None:
                default_val = row[self._default_category]
                if default_val in self._default_map:
                    results.append(self._default_map[default_val])
                    used_default = True

            if not used_default:
                results.append(np.nan)
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} missing groups were not present in the training data.")

        return np.array(results)
