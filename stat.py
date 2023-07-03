df = pd.read_csv("water_quality.csv")  # load your dataset
print(df.describe())

from scipy import stats

# Define a list of columns for which you want to check normality
columns = ['DO', 'BOD', 'pH', 'Turbidity', 'Nitrate', 'Phosphate']

# Loop over the columns and perform Shapiro-Wilk test
for col in columns:
    stat, p_value = stats.shapiro(df[col])
    print(f"\nShapiro-Wilk Test for {col}:\nStatistic={stat}, p-value={p_value}")
    if p_value > 0.05:
        print(f'{col} looks Gaussian (fail to reject H0)')
    else:
        print(f'{col} does not look Gaussian (reject H0)')

from statsmodels.tsa.stattools import adfuller

# Define a list of columns for which you want to check stationarity
columns = ['DO', 'BOD', 'pH', 'Turbidity', 'Nitrate', 'Phosphate']

# Loop over the columns and perform Augmented Dickey-Fuller test
for col in columns:
    result = adfuller(df[col])
    print(f"\nAugmented Dickey-Fuller Test for {col}:\nADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")

    if result[1] > 0.05:
        print(f'{col} is not stationary (fail to reject H0)')
    else:
        print(f'{col} is stationary (reject H0)')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Let's assume 'df' is your DataFrame and it has no missing values. If there are missing values, you need to handle them. For example, you can fill missing values with the mean of the column:
df.fillna(df.mean(), inplace=True)

# Specify the data columns (features) and the target variable
X = df.drop('target_variable', axis=1)
y = df['target_variable']

# Normalize the data (features)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression

# Create a Linear Regression model object
regression_model = LinearRegression()

# Train the model using the training sets
regression_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_MLR = regression_model.predict(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu')) # Input layer
model.add(Dense(10, activation='relu')) # Hidden layer
model.add(Dense(1)) # Output layer

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10)

# Make predictions using the testing set
y_pred_ANN = model.predict(X_test)

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor object
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training sets
rf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_RF = rf.predict(X_test)
