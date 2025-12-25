import lightgbm as lgb


def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """
    Train a LightGBM regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target values.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    # Prepare LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train)
    test_dataset = lgb.Dataset(
        X_test,
        label=y_test,
        reference=train_dataset
    )

    # LightGBM parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }

    # Train with early stopping
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[train_dataset, test_dataset],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(50),
        ],
    )

    return model