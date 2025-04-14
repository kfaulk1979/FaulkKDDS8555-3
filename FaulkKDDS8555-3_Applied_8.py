import numpy as np
np.random.seed(1)  # for reproducibility

n = 100
# Generate predictor X and noise vector epsilon:
X = np.random.normal(0, 1, n)
epsilon = np.random.normal(0, 1, n)

beta0, beta1, beta2, beta3 = 5, 2, -3, 0.5
Y = beta0 + beta1 * X + beta2 * X**2 + beta3 * X**3 + epsilon

import statsmodels.api as sm
import itertools

# Prepare candidate predictors:
X_candidates = np.column_stack([X**power for power in range(1, 11)])
predictor_names = [f"X^{power}" for power in range(1, 11)]

# Function to compute Cp:
def compute_cp(model, sigma2_hat, n, p):
    # Cp = RSS/hat(sigma^2) - (n - 2p)
    RSS = sum(model.resid**2)
    cp = RSS / sigma2_hat - (n - 2 * p)
    return cp

# Fit full model to get an estimator for sigma^2
X_full = sm.add_constant(X_candidates)
full_model = sm.OLS(Y, X_full).fit()
sigma2_hat = full_model.mse_resid



def forward_stepwise(X_candidates, Y, predictor_names):
    n, p_total = X_candidates.shape
    remaining = list(range(p_total))
    selected = []
    current_score, current_model = float('inf'), None
    models = {}
    selected_history = {}

    for i in range(p_total):
        scores_with_candidates = []
        for candidate in remaining:
            predictors = selected + [candidate]
            X_model = sm.add_constant(X_candidates[:, predictors])
            model = sm.OLS(Y, X_model).fit()
            cp = compute_cp(model, sigma2_hat, n, len(predictors)+1)  # +1 for intercept
            scores_with_candidates.append((cp, candidate, model))
        
        # Choose the candidate with the smallest Cp
        scores_with_candidates.sort(key=lambda x: x[0])
        best_cp, best_candidate, best_model = scores_with_candidates[0]
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        models[tuple(sorted(selected))] = (best_cp, best_model)
        selected_history[tuple(sorted(selected))] = list(selected)  # Save copy of list
        print(f"Step {i+1}: Added {predictor_names[best_candidate]}, Cp = {best_cp:.2f}")
    
    # Choose the best model and return its predictors too
    best_predictor_tuple = min(models.keys(), key=lambda k: models[k][0])
    best_model_forward = models[best_predictor_tuple][1]
    best_predictors = selected_history[best_predictor_tuple]
    
    print("\nFinal forward stepwise model coefficients:")
    print(best_model_forward.params)
    
    print("\nPredictors used in final model:")
    print([predictor_names[i] for i in best_predictors])
    
    return best_model_forward, best_predictors

predictor_names = [f"X^{i+1}" for i in range(X_candidates.shape[1])]
models_forward = forward_stepwise(X_candidates, Y, predictor_names)

best_model_forward, best_predictors = forward_stepwise(X_candidates, Y, predictor_names)

# Safe to use right here
print("\nFinal forward stepwise model coefficients:")
print(best_model_forward.params)

print("\nPredictors used in final model:")
print([predictor_names[i] for i in best_predictors])


def backward_stepwise(X_candidates, Y):
    n, p_total = X_candidates.shape
    selected = list(range(p_total))  # start with all predictors
    models = {}
    
    # Fit initial full model:
    X_model = sm.add_constant(X_candidates[:, selected])
    current_model = sm.OLS(Y, X_model).fit()
    current_cp = compute_cp(current_model, sigma2_hat, n, len(selected)+1)
    models[tuple(sorted(selected))] = (current_cp, current_model)
    print(f"Start: Full model with predictors {[predictor_names[i] for i in selected]}, Cp = {current_cp:.2f}")
    
    for i in range(p_total-1):
        scores_after_removal = []
        for candidate in selected:
            new_predictors = [p for p in selected if p != candidate]
            X_model = sm.add_constant(X_candidates[:, new_predictors])
            model = sm.OLS(Y, X_model).fit()
            cp = compute_cp(model, sigma2_hat, n, len(new_predictors)+1)
            scores_after_removal.append((cp, candidate, model))
        scores_after_removal.sort(key=lambda x: x[0])
        best_cp, removed, best_model = scores_after_removal[0]
        selected.remove(removed)
        models[tuple(sorted(selected))] = (best_cp, best_model)
        print(f"Step {i+1}: Removed {predictor_names[removed]}, Cp = {best_cp:.2f}")
    return models

models_backward = backward_stepwise(X_candidates, Y)
best_model_backward = min(models_backward.values(), key=lambda x: x[0])[1]
print("\nFinal backward stepwise model coefficients:")
print(best_model_backward.params)
selected_predictors = best_model_backward.model.exog_names
print("Predictors selected in the final backward stepwise model:")
for predictor in selected_predictors:
    print(predictor)

import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# Create predictors with X, X^2, ..., X^10 (already defined as X_candidates)
# LassoCV will automatically perform cross-validation.
lasso_cv = LassoCV(cv=5, max_iter=500000, random_state=1, alphas=np.logspace(-4, 1, 100)).fit(X_candidates, Y)

# Plotting the cross-validation error curve.
m_log_alphas = -np.log10(lasso_cv.alphas_)
plt.figure(figsize=(8, 6))
plt.plot(m_log_alphas, np.mean(lasso_cv.mse_path_, axis=1), marker='o', linestyle='--')
plt.xlabel('-log(alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso CV: MSE vs. -log(alpha)')
plt.show()

print("Optimal alpha (lambda):", lasso_cv.alpha_)
coef_names = ["Intercept"] + predictor_names
coefficients = np.concatenate(([lasso_cv.intercept_], lasso_cv.coef_))
print("\nLasso Coefficient Estimates:")
for name, coef in zip(coef_names, coefficients):
    print(f"{name}: {coef:.3f}")


# Generate new response vector using only X^7
beta0 = 5
beta7 = 3  # chosen constant
Y_new = beta0 + beta7 * X**7 + epsilon

# --- Forward Stepwise on new Y ---
print("\nForward Stepwise with Y = beta0 + beta7 * X^7 + error")
models_forward_new = forward_stepwise(X_candidates, Y_new, predictor_names)
best_model_forward_new = best_model_forward_new = models_forward_new[0]
print("Selected forward stepwise coefficients:")
print(best_model_forward_new.params)

# --- Lasso on new Y ---
lasso_cv_new = LassoCV(cv=5, max_iter=500000, random_state=1, alphas=np.logspace(-4, 1, 100)).fit(X_candidates, Y_new)

plt.figure(figsize=(8, 6))
m_log_alphas_new = -np.log10(lasso_cv_new.alphas_)
plt.plot(m_log_alphas_new, np.mean(lasso_cv_new.mse_path_, axis=1), marker='o', linestyle='--')
plt.xlabel('-log(alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso CV with new Y: MSE vs. -log(alpha)')
plt.show()

print("Optimal alpha (lambda) for new Y:", lasso_cv_new.alpha_)
coef_names = ["Intercept"] + predictor_names
coefficients_new = np.concatenate(([lasso_cv_new.intercept_], lasso_cv_new.coef_))
print("\nLasso Coefficient Estimates for new Y:")
for name, coef in zip(coef_names, coefficients_new):
    print(f"{name}: {coef:.3f}")

