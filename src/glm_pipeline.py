import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, target_col, feature_cols):
    X = df[feature_cols]
    X = sm.add_constant(X)
    y = df[target_col]
    return X, y

def fit_glm(X, y, family=sm.families.Binomial()):
    model = sm.GLM(y, X, family=family)
    results = model.fit()
    return results

def plot_residuals(results, X, y, save_path=None):
    """Plot residual diagnostics for the GLM model."""
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GLM Model Diagnostics', fontsize=16)
    
    # Get fitted values and residuals
    fitted_values = results.fittedvalues
    residuals = results.resid_pearson
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Pearson Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q Plot
    sm.qqplot(residuals, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # 3. Scale-Location Plot
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    axes[1, 0].scatter(fitted_values, sqrt_abs_residuals, alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Residuals|')
    axes[1, 0].set_title('Scale-Location Plot')
    
    # 4. Residuals vs Leverage
    leverage = results.get_influence().hat_matrix_diag
    axes[1, 1].scatter(leverage, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Pearson Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_distributions(df, feature_cols, target_col, save_path=None):
    """Plot distributions of features by target variable."""
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'Feature Distributions by {target_col}', fontsize=16)
    
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_features > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        
        # Create histograms for each target class
        for target_val in df[target_col].unique():
            subset = df[df[target_col] == target_val][feature]
            ax.hist(subset, alpha=0.7, label=f'{target_col}={target_val}', bins=20)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df, feature_cols, save_path=None):
    """Plot correlation matrix of features."""
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_coefficients(results, feature_names, save_path=None):
    """Plot model coefficients with confidence intervals."""
    params = results.params
    conf_int = results.conf_int()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(feature_names))
    
    # Plot coefficients
    ax.barh(y_pos, params, alpha=0.7)
    
    # Plot confidence intervals
    for i, (param, (lower, upper)) in enumerate(zip(params, conf_int.values)):
        ax.plot([lower, upper], [i, i], 'k-', linewidth=2)
        ax.plot([lower, lower], [i-0.1, i+0.1], 'k-', linewidth=2)
        ax.plot([upper, upper], [i-0.1, i+0.1], 'k-', linewidth=2)
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Model Coefficients with 95% Confidence Intervals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and preprocess data
    df = load_data("data/sample.csv")
    feature_cols = ["age", "income", "balance"]
    target_col = "default"
    
    X, y = preprocess_data(df, target_col, feature_cols)
    
    # Create visualizations before fitting the model
    print("Creating exploratory data visualizations...")
    plot_feature_distributions(df, feature_cols, target_col)
    plot_correlation_matrix(df, feature_cols)
    
    # Fit the GLM model
    results = fit_glm(X, y)
    print(results.summary())
    
    # Create model diagnostic plots
    print("Creating model diagnostic plots...")
    plot_residuals(results, X, y)
    
    # Plot coefficients
    feature_names = ['const'] + feature_cols
    plot_coefficients(results, feature_names)
    
    # Save the model
    results.save("models/glm_model.pickle")

if __name__ == "__main__":
    main()
