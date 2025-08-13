import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

def main():
    df = load_data("data/sample.csv")
    X, y = preprocess_data(df, "default", ["age", "income", "balance"])
    results = fit_glm(X, y)
    print(results.summary())
    results.save("models/glm_model.pickle")

if __name__ == "__main__":
    main()
