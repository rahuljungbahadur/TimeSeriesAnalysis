# time_series_streamlit_playground.py
# Streamlit app for time-series specific exploration

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import seaborn as sns
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import f_regression

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def simulate_ts(n:int, seed:int,
                rho:float,
                coef_x1:float, coef_x2:float, coef_x3:float,
                coef_x4:float, eps_sigma:float,
                x_corr_within:float = 0.0,
                x3_is_noise:bool = False,
                x4_depends_on_y:bool = False):
    """Simulate time series where y_t depends on y_{t-1} and lagged Xs."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = x_corr_within * X1 + np.sqrt(max(1 - x_corr_within**2, 0)) * rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)
    if x3_is_noise:
        X3 = rng.normal(0, 1, n)
    X4 = rng.normal(0, 1, n)

    y = np.zeros(n)
    for t in range(1, n):
        # lagged Xs
        x1_l, x2_l, x3_l, x4_l = X1[t-1], X2[t-1], X3[t-1], X4[t-1]
        # base DGP
        y[t] = rho * y[t-1] + coef_x1 * x1_l + coef_x2 * x2_l + coef_x3 * x3_l + coef_x4 * x4_l + rng.normal(0, eps_sigma)
        if x4_depends_on_y:
            X4[t] = 0.7 * y[t] + 0.3 * rng.normal(0, 1)  # leakage example

    df = pd.DataFrame({
        "y": y[1:],
        "y_lag": y[:-1],
        "X1_lag": X1[:-1],
        "X2_lag": X2[:-1],
        "X3_lag": X3[:-1],
        "X4_lag": X4[:-1],
    })
    return df

def fit_model(X:pd.DataFrame, y:np.ndarray, model_type:str, test_size:float, seed:int):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1)
    else:
        model = LinearRegression()
    model.fit(Xtr, ytr)
    pred_tr = model.predict(Xtr)
    pred_te = model.predict(Xte)
    metrics = {
        "train_MSE": float(mean_squared_error(ytr, pred_tr)),
        "test_MSE": float(mean_squared_error(yte, pred_te)),
        "train_R2": float(r2_score(ytr, pred_tr)),
        "test_R2": float(r2_score(yte, pred_te)),
    }
    return model, metrics, (Xtr, Xte, ytr, yte)

def plot_y_correlations(X_df: pd.DataFrame, y: np.ndarray, title: str):
    st.subheader(title)
    fig, axes = plt.subplots(nrows=1, ncols=len(X_df.columns), figsize=(5 * len(X_df.columns), 4))
    if len(X_df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, X_df.columns):
        ax.scatter(X_df[col], y, alpha=0.6)
        r = np.corrcoef(X_df[col], y)[0, 1]
        ax.annotate(f"ρ={r:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("y")
    plt.tight_layout()
    st.pyplot(fig)

def plot_both_importances(model, Xte, yte, title_prefix: str):
    if hasattr(model, "feature_importances_"):
        native_vals = np.asarray(model.feature_importances_, dtype=float)
        native_label = "Model importance"
    elif hasattr(model, "coef_"):
        coef = np.ravel(np.asarray(model.coef_))
        native_vals = coef
        native_label = "Coefficient (signed)"
    else:
        native_vals = None
    perm = permutation_importance(model, Xte, yte, n_repeats=20, random_state=0, n_jobs=-1)
    perm_means = perm.importances_mean
    perm_stds = perm.importances_std
    if native_vals is None:
        plot_perm_importance(model, Xte, yte, title=f"{title_prefix} — Permutation")
        return
    cols = np.array(Xte.columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    order_n = np.argsort(native_vals if hasattr(model, "feature_importances_") else np.abs(native_vals))
    axes[0].barh(cols[order_n], native_vals[order_n])
    if not hasattr(model, "feature_importances_"):
        axes[0].axvline(0, color='k', linewidth=1)
    axes[0].set_title(f"{title_prefix} — Native")
    axes[0].set_xlabel(native_label)
    order_p = np.argsort(perm_means)
    axes[1].barh(cols[order_p], perm_means[order_p], xerr=perm_stds[order_p])
    axes[1].set_title(f"{title_prefix} — Permutation")
    axes[1].set_xlabel("Permutation importance (mean ± sd)")
    plt.tight_layout()
    st.pyplot(fig)

def plot_perm_importance(model, Xte, yte, title: str):
    perm = permutation_importance(model, Xte, yte, n_repeats=20, random_state=0, n_jobs=-1)
    means = perm.importances_mean
    stds = perm.importances_std
    order = np.argsort(means)
    fig, ax = plt.subplots()
    ax.barh(np.array(Xte.columns)[order], means[order], xerr=stds[order])
    ax.set_xlabel("Permutation importance (mean ± sd)")
    ax.set_title(title)
    st.pyplot(fig)

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="TS Mediators & Proxies Playground", layout="wide")
st.title("🔬 Time Series: Mediators, Autocorrelation & Importances")

with st.sidebar:
    st.header("Simulation Controls")
    n = st.slider("n (samples)", 200, 5000, 1000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
    eps_sigma = st.slider("Noise σ (epsilon)", 0.05, 2.0, 0.5, step=0.05)
    model_type = st.selectbox("Model", ["RandomForest", "LinearRegression"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, step=0.05)

    st.divider()
    rho = st.slider("ρ (autocorrelation)", 0.0, 0.95, 0.5, 0.05)
    coef_x1 = st.slider("coef(X1_lag)", 0.0, 2.0, 0.8, 0.05)
    coef_x2 = st.slider("coef(X2_lag)", 0.0, 2.0, 0.7, 0.05)
    coef_x3 = st.slider("coef(X3_lag)", 0.0, 2.0, 0.3, 0.05)
    coef_x4 = st.slider("coef(X4_lag)", 0.0, 2.0, 0.0, 0.05)
    x_corr_within = st.slider("Corr(X2, X1)", 0.0, 0.95, 0.8, 0.05)
    x4_depends_on_y = st.checkbox("Leakage: X4 depends on y", value=False)
    include_set = st.selectbox("Features to feed model", ["Lagged Xs only","Lagged Xs + y_lag","y_lag only"])

# ------------------------------
# Run simulation & fit
# ------------------------------
sim_kwargs = dict(n=n, seed=seed, coef_x1=coef_x1, coef_x2=coef_x2, coef_x3=coef_x3,
                  coef_x4=coef_x4, eps_sigma=eps_sigma, x_corr_within=x_corr_within,
                  x3_is_noise=False, x4_depends_on_y=x4_depends_on_y)
df = simulate_ts(rho=rho, **sim_kwargs)
st.subheader("Sample of simulated data")
st.dataframe(df.head())

# Prepare data
y = df.pop("y").values
if include_set == "Lagged Xs only":
    X = df[["X1_lag", "X2_lag", "X3_lag", "X4_lag"]]
elif include_set == "Lagged Xs + y_lag":
    X = df[["X1_lag", "X2_lag", "X3_lag", "X4_lag", "y_lag"]]
else:
    X = df[["y_lag"]]

# Fit model
model, metrics, (Xtr, Xte, ytr, yte) = fit_model(X, y, model_type, test_size, seed)
st.subheader("Metrics")
st.write(pd.DataFrame([metrics]))

st.subheader("Correlation heatmap")
df_corr = Xte.copy()
df_corr = df_corr.assign(y=yte)
plot_y_correlations(Xte, yte, "Feature–y correlations (test set)")

st.subheader("Permutation importance (test set)")
plot_both_importances(model, Xte, yte, title_prefix=f"{model.__class__.__name__}")
