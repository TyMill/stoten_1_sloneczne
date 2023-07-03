# main.py
from libraries import *
import fut_importance as fi
import stat
import vis

# Load the data
print("Loading the data...")
data = pd.read_csv('water_quality.csv')

# Perform descriptive statistics, normality tests, and stationarity tests
print("Performing data analysis...")
stat.descriptive_statistics(data)
stat.normality_tests(data)
stat.stationarity_tests(data)

# Generate correlation matrices, time series plots, and heatmaps
print("Creating visualizations...")
vis.correlation_matrix(data)
vis.time_series_plots(data)
vis.heatmap(data)

# Prepare the data for machine learning models
print("Preprocessing the data...")
X, y = data.drop('Quality Index', axis=1), data['Quality Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning models and evaluate their performance
print("Training and evaluating models...")
models = [LinearRegression(), RandomForestRegressor(), MLPRegressor()]
model_names = ['Multiple Linear Regression', 'Random Forest', 'Artificial Neural Networks']
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Performance of {name}:')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R^2 score: {r2_score(y_test, y_pred)}')

    # Analyze the feature importance of each model and interpret the results
    print(f'Feature importance for {name}:')
    fi.feature_importance(model, X.columns)

# If possible, use the best-performing model to predict future water quality
# and generate visualizations to show model performance, feature importance, and future predictions
# (Add your own code here...)

print("All tasks completed.")
