from xgboost import XGBRegressor
import time
def i_feel_lucky_xgboost_training(train_df, test_df, features, target, name,
                                  n_estimators=80, max_depth=4, learning_rate=0.05):
    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    xgb_clf = XGBRegressor(base_score=0.5, booster='gbtree',
                            colsample_bylevel=1, colsample_bynode=1,
                            colsample_bytree=1, gamma=0,
                            learning_rate=learning_rate, max_delta_step=0,
                            max_depth=max_depth, min_child_weight=1,
                            missing=None, n_estimators=n_estimators, n_jobs=1,
                            nthread=None, objective='reg:squarederror',
                            random_state=0, reg_alpha=0, reg_lambda=1,
                            scale_pos_weight=1, seed=1, silent=None,
                            subsample=0.5, verbosity=1)
    start = time.time()
    xgb_clf.fit(x_train, y_train.values.ravel())
    end = time.time()
    clf_name = name
    test_df[clf_name] = xgb_clf.predict(x_test)#[:, 1]
    return xgb_clf, clf_name