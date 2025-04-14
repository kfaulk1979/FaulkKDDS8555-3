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

# PCR
X_scaled = preprocessor.fit_transform(X)
X_test_scaled = preprocessor.transform(X_test)

# PCA: 95% variance retained
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_pca_train, X_pca_val, y_pca_train, y_pca_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)
pcr_model = LinearRegression()
pcr_model.fit(X_pca_train, y_pca_train)

# Get regression coefficients for principal components
pcr_component_coefs = pcr_model.coef_

# Display component coefficients
print("Coefficients for Principal Components (PCR):")
for i, coef in enumerate(pcr_component_coefs):
    print(f"PC{i+1}: {coef:.4f}")

# Convert component coefficients back to original feature space
original_feature_influence = np.dot(pca.components_.T, pcr_model.coef_)

# Get feature names from preprocessor
encoded_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(['Sex'])
all_features = numerical + list(encoded_feature_names)

# Create DataFrame of mapped coefficients
influence_df = pd.DataFrame({
    'Feature': all_features,
    'Influence': original_feature_influence
})

# Sort and display
influence_df = influence_df.sort_values(by='Influence', key=abs, ascending=False)
print("\n Approximated Influence of Original Predictors in PCR:")
print(influence_df)

# Predict and evaluate
pcr_val_preds = pcr_model.predict(X_pca_val)
pcr_rmse = np.sqrt(mean_squared_error(y_pca_val, pcr_val_preds))
print(f"PCR RMSE: {pcr_rmse:.3f}")

# Residual diagnostics for PCR
residuals_pcr = y_pca_val - pcr_val_preds

# Residuals vs. Predicted
plt.scatter(pcr_val_preds, residuals_pcr, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("PCR: Residuals vs. Predicted")
plt.xlabel("Predicted Rings")
plt.ylabel("Residuals")
plt.show()

# Q-Q Plot
stats.probplot(residuals_pcr, dist="norm", plot=plt)
plt.title("PCR: Q-Q Plot")
plt.show()

# Final PCR predictions for test set
pcr_predictions = pcr_model.predict(X_test_pca)

# Round predictions to nearest integer
pcr_predictions_rounded = np.round(pcr_predictions).clip(min=0).astype(int)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Rings': pcr_predictions_rounded
})

# Save to CSV
submission_path = "/Users/kevinfaulk/Documents/DDS-8555/abalone_submission_pcr.csv"
submission.to_csv(submission_path, index=False)
print(f"Rounded submission file saved: {submission_path}")

