# MLTT assignment 1
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load a smaller subset of the California Housing dataset for faster testing
data = fetch_california_housing()
X, _, y, _ = train_test_split(data.data, data.target, test_size=0.9, random_state=42)

# Split the smaller subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with feature selection and regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  # Set k to 'all' for all features
    ('regressor', RandomForestRegressor())
])

# Define hyperparameters for RandomizedSearchCV
param_dist = {
    'feature_selection__k': [5, 8, 'all'],  # Adjust the values based on the number of features in your dataset
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

# Use RandomizedSearchCV for hyperparameter tuning with fewer iterations
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, error_score='raise')

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", random_search.best_params_)

# Evaluate the model on the test set
y_pred = random_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Best Mean Squared Error from Cross-Validation: ", -random_search.best_score_)  # Best MSE from cross-validation
print("Root Mean Squared Error on Test Set: ",rmse)
