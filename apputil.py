import numpy as np
import pandas as pd

class GroupEstimate:
    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self._group_map = None        # mapping from group-tuple -> estimate value
        self._cols = None             # list of column names used for grouping
        self._default_category = None # name of default category column (or None)
        self._default_map = None      # mapping from single-category-value -> estimate

    def fit(self, X, y, default_category=None):
        """
        X : pandas DataFrame of categorical columns (or convertible to one)
        y : 1-D array-like of numeric values (no missing values expected by spec)
        default_category : optional str, name of the column to fall back on when a full-group is missing
        """
        # normalize inputs
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = pd.Series(y).reset_index(drop=True)

        if len(X_df) != len(y_ser):
            raise ValueError("X and y must have the same length")

        # save columns used for grouping
        self._cols = list(X_df.columns)
        self._default_category = default_category

        # combine and group by the grouping columns; use observed=True to ignore unobserved categories
        combined = X_df.copy().reset_index(drop=True)
        combined["_y"] = y_ser.values

        grouped = combined.groupby(self._cols, observed=True)
        agg_func = np.mean if self.estimate == "mean" else np.median
        group_stats = grouped["_y"].agg(agg_func)

        # Build mapping from tuple-of-values -> numeric estimate
        group_map = {}
        for key, val in group_stats.items():
            # ensure key is a tuple of length len(self._cols)
            if not isinstance(key, tuple):
                key = (key,)
            group_map[tuple(key)] = float(val)

        self._group_map = group_map

        # If default_category provided, build mapping from that single column -> estimate
        if default_category is not None:
            if default_category not in self._cols:
                raise ValueError("default_category must be one of the columns in X")
            # group by the default column only, compute same agg
            default_grouped = combined.groupby(default_category, observed=True)["_y"].agg(agg_func)
            default_map = {}
            for k, v in default_grouped.items():
                default_map[k] = float(v)
            self._default_map = default_map
        else:
            self._default_map = None

        return self

    def predict(self, X_):
        """
        X_ : array-like or DataFrame with same number of columns (and same column names if DataFrame)
        Returns a list of estimates (float) with np.nan for unknown groups.
        Prints a message indicating how many missing groups were encountered.
        """
        if self._group_map is None or self._cols is None:
            raise ValueError("Model has not been fitted. Call fit(X, y) before predict().")

        # normalize X_ into DataFrame
        if isinstance(X_, pd.DataFrame):
            Xp = X_.copy().reset_index(drop=True)
            # ensure columns align with fitted columns
            if list(Xp.columns) != self._cols:
                # if same columns but different order, reorder
                if set(Xp.columns) == set(self._cols):
                    Xp = Xp[self._cols]
                else:
                    raise ValueError("Predict input columns must match the columns used in fit")
        else:
            # assume array-like of rows; if shape is (n_rows, n_cols) or list of lists
            Xp = pd.DataFrame(X_, columns=self._cols)

        results = []
        missing_count = 0

        for _, row in Xp.iterrows():
            key = tuple(row[col] for col in self._cols)
            if key in self._group_map:
                results.append(self._group_map[key])
                continue

            # try default category fallback if available
            used_default = False
            if self._default_map is not None:
                default_val = row[self._default_category]
                if default_val in self._default_map:
                    results.append(self._default_map[default_val])
                    used_default = True

            if not used_default:
                results.append(float("nan"))
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} missing groups were not present in the training data.")

        return results
