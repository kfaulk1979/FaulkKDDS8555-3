import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/test.csv")

#Set up features and target
X = train.drop(columns=['id', 'Rings'])
y = train['Rings']
X_test = test.drop(columns=['id'])

#Define column types
categorical = ['Sex']
numerical = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

#Preprocessing: scale numerics, one-hot encode categoricals
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first'), categorical)
])

#Build Lasso pipeline with preprocessing
lasso_pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', LassoCV(cv=5, random_state=42))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
lasso_pipeline.fit(X_train, y_train)

#Predict and evaluate
val_preds = lasso_pipeline.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Lasso RMSE: {lasso_rmse:.3f}")

#Residual diagnostics for Lasso
residuals_lasso = y_val - val_preds

# Residuals vs. Predicted
plt.scatter(val_preds, residuals_lasso, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Lasso: Residuals vs. Predicted")
plt.xlabel("Predicted Rings")
plt.ylabel("Residuals")
plt.show()

# Q-Q Plot
stats.probplot(residuals_lasso, dist="norm", plot=plt)
plt.title("Lasso: Q-Q Plot")
plt.show()

# Train Lasso on all data for submission
lasso_pipeline.fit(X, y)
lasso_predictions = lasso_pipeline.predict(X_test)


#Extract feature names after preprocessing
encoded_feature_names = lasso_pipeline.named_steps['pre'].transformers_[1][1].get_feature_names_out(['Sex'])
all_features = numerical + list(encoded_feature_names)

#Get Lasso model and coefficients
lasso_model = lasso_pipeline.named_steps['model']
coefficients = lasso_model.coef_

#Combine into DataFrame
coef_df = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': coefficients
})

#Display only non-zero coefficients
nonzero_coefs = coef_df[coef_df['Coefficient'] != 0]
print("Non-zero Lasso Coefficients:")
print(nonzero_coefs.sort_values(by='Coefficient', key=abs, ascending=False))


# Round predictions to nearest integer
lasso_predictions_rounded = np.round(lasso_predictions).clip(min=0).astype(int)


# Create submission DataFrame with rounded values
submission = pd.DataFrame({
    'id': test['id'],
    'Rings': lasso_predictions_rounded
})

submission.to_csv("/Users/kevinfaulk/Documents/DDS-8555/abalone_submission_lasso.csv", index=False)
print("Submission file saved as abalone_submission_lasso.csv")

