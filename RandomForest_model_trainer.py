from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV


def RandomForest_model_trainer(X_train, y_train):
    #cross-validation
    model = RandomForestClassifier(random_state=0)
    cross_val = cross_val_score(model, X_train, y_train, cv=5)
    print( "%f%% is the result for the first model\n" % (cross_val.mean()))

    #We now tune parameters
    n_estimators = [10, 100, 500, 1000]
    max_depth = [None, 5, 10, 20]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    
    #We now want to use grid search
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        verbose=2,
                        n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    print("\nThe best result was with")
    print(grid_result.best_params_)
    print("and had a precision of %f%%, this is a improvement of %f%%\n"
          %(grid_result.best_score_, grid_result.best_score_-cross_val.mean()))

    first_opt = grid_result.best_score_

    #Now we optimize leaf size
    leaf_samples = [1, 2, 3, 4, 5, 6]
    param_grid = dict(min_samples_leaf=leaf_samples)

    model = grid_result.best_estimator_
 
    #grid search
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        verbose=2,
                        n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    print("\nThe best result was with")
    print(grid_result.best_params_)
    print("and had a precision of %f%%, this is a improvement of %f%%\n"
          %(grid_result.best_score_, grid_result.best_score_-first_opt))

    second_opt = grid_result.best_score_

    max_features = [5, 8, 12, None]
    bootstrap = [True, False]
    param_grid = dict(max_features=max_features,bootstrap=bootstrap)

    model = grid_result.best_estimator_

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        verbose=2,
                        n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    print("\nThe best result was with")
    print(grid_result.best_params_)
    print("and had a precision of %f%%, this is a improvement of %f%%\n"
          %(grid_result.best_score_, grid_result.best_score_-second_opt))

    #We are not ready to predict the test data
    model = grid_result.best_estimator_
    #cross-validating best model
    print("result of best model cross-validated it: %f%%.\n" %(cross_val_score(model, X_train, y_train, cv=5).mean()))
    print("Success model optimized!!!")
    
    return model





