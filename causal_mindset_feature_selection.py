# app.py
# Streamlit playground for mediators, proxies, autocorrelation, and importances
# Includes: correlation heatmap + X4 visibility fix

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


# ------------------------------
# Utilities
# ------------------------------
def make_fig():
    fig = plt.figure()
    return fig

@st.cache_data(show_spinner=False)
def simulate_non_ts(n:int, seed:int,
                    a_x1_x2:float, b_x1_x3:float, c_x1_noise:float,
                    beta_x1:float, beta_x4:float, eps_sigma:float,
                    add_x4:bool, non_linear_x1:bool = False, non_linear_x4:bool = False):
    rng = np.random.default_rng(seed)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)
    X1 = a_x1_x2 * X2 + b_x1_x3 * X3 + c_x1_noise * rng.normal(0, 1, n)
    if add_x4:
        X4 = rng.normal(0, 1, n)
    else:
        X4 = np.zeros(n)

    if non_linear_x1:
        y = beta_x1 * np.sin(X1) + (beta_x4 * np.sin(X4) if add_x4 and non_linear_x4 else beta_x4 * X4) + rng.normal(0, eps_sigma, n)
    elif non_linear_x4:
        y = beta_x1 * X1 + beta_x4 * np.sin(X4) + rng.normal(0, eps_sigma, n)
    else:
        y = beta_x1 * X1 + (beta_x4 * X4 if add_x4 else 0.0) + rng.normal(0, eps_sigma, n)

    df = pd.DataFrame({"y": y, "X1": X1, "X2": X2, "X3": X3})
    if add_x4:
        df["X4"] = X4
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



# --- Data simulation example ---
# Example functional forms (printed after sample generation):
# Non‑TS: y = β₁ · X1 + β₄ · X4 + ε, with X1 = a · X2 + b · X3 + c · ν

def print_functional_form_non_ts(beta_x1: float, add_x4: bool, beta_x4: float):
    st.markdown("<div style='text-align: center; font-weight: bold; font-size: 1.2em;'>Functional form (Non‑TS):</div>", unsafe_allow_html=True)
    if add_x4:
        st.latex(
            fr"y = {beta_x1} \cdot X_1 + {beta_x4} \cdot X_4 + \epsilon"
        )
    else:
        st.latex(
            fr"y = {beta_x1} \cdot X_1 + \epsilon"
        )
    st.latex(r"X_1 = a \cdot X_2 + b \cdot X_3 + c \cdot \nu")

# --- Convenience hooks to wire into your app right after sample data tables ---

def show_after_sample_non_ts(df_head: pd.DataFrame, *, beta_x1: float, add_x4: bool, beta_x4: float):
    st.subheader("Sample of simulated data")
    st.dataframe(df_head)
    print_functional_form_non_ts(beta_x1, add_x4, beta_x4)


# --- Plotting helpers ---

from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def plot_feature_selection_non_ts(Xtr: pd.DataFrame, ytr: np.ndarray, method_rfe: bool = True):
    """
    Visualize feature selection signals for NON-time-series:
      - ANOVA F-test (f_regression)
      - Mutual information (mutual_info_regression)
      - LassoCV coefficients (signed)
      - Optional: RFE ranking converted to a score (higher=better)

    Uses TRAINING data only to avoid information leakage.
    """
    cols = np.array(Xtr.columns)

    # 1) ANOVA F-test
    f_scores, f_pvals = f_regression(Xtr, ytr)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # 2) Mutual information (random_state for reproducibility)
    mi_scores = mutual_info_regression(Xtr, ytr, random_state=0)

    # 3) LassoCV (with scaling)
    lasso = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                          LassoCV(cv=5, random_state=0, n_alphas=100, max_iter=5000))
    lasso.fit(Xtr, ytr)
    # Extract coefficients from the final step
    lasso_coef = lasso.named_steps["lassocv"].coef_

    # 4) (Optional) RFE with LinearRegression to get a ranking
    rfe_score = None
    if method_rfe:
        lin = LinearRegression()
        rfe = RFE(estimator=lin, n_features_to_select=1, step=1)
        rfe.fit(Xtr, ytr)
        # Convert ranking (1=best) to a score where higher is better
        max_rank = np.max(rfe.ranking_)
        rfe_score = (max_rank + 1 - rfe.ranking_).astype(float)

    # --- Plotting ---
    n_panels = 4 if method_rfe else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))

    # Panel 1: ANOVA F
    order_f = np.argsort(f_scores)
    axes[0].barh(cols[order_f], f_scores[order_f])
    axes[0].set_title("ANOVA F (train)")
    axes[0].set_xlabel("F score")

    # Panel 2: Mutual Information
    order_mi = np.argsort(mi_scores)
    axes[1].barh(cols[order_mi], mi_scores[order_mi])
    axes[1].set_title("Mutual Information (train)")
    axes[1].set_xlabel("MI")

    # Panel 3: LassoCV coefficients (signed)
    order_lasso = np.argsort(np.abs(lasso_coef))
    axes[2].barh(cols[order_lasso], lasso_coef[order_lasso])
    axes[2].axvline(0, color="k", linewidth=1)
    axes[2].set_title("LassoCV coefficients (train)")
    axes[2].set_xlabel("Coef (signed)")

    # Panel 4: RFE score (optional)
    if method_rfe:
        order_rfe = np.argsort(rfe_score)
        axes[3].barh(cols[order_rfe], rfe_score[order_rfe])
        axes[3].set_title("RFE score (train)")
        axes[3].set_xlabel("Higher = better")

    plt.tight_layout()
    st.pyplot(fig)

def plot_threshold_selectors_non_ts(Xtr: pd.DataFrame, ytr: np.ndarray, alpha: float = 0.05, k: int | None = None):
    """
    Visualize scikit-learn's threshold-based feature selectors on TRAIN data:
      - SelectKBest
      - SelectFPR
      - SelectFDR
      - SelectFWE
    All using f_regression for comparability (provides scores + p-values).
    Shows the common ANOVA F scores and highlights which features each method selects.
    """
    cols = np.array(Xtr.columns)
    # Compute F and p once
    f_scores, p_vals = f_regression(Xtr, ytr)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)

    if k is None:
        k = max(1, int(np.ceil(Xtr.shape[1] / 2)))

    selectors = [
        ("SelectKBest", SelectKBest(score_func=f_regression, k=k)),
        ("SelectFPR",  SelectFpr(score_func=f_regression, alpha=alpha)),
        ("SelectFDR",  SelectFdr(score_func=f_regression, alpha=alpha)),
        ("SelectFWE",  SelectFwe(score_func=f_regression, alpha=alpha)),
    ]

    # Sort once by F for a consistent visual order
    order = np.argsort(f_scores)
    ordered_cols = cols[order]
    ordered_scores = f_scores[order]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for ax, (name, selector) in zip(axes, selectors):
        selector.fit(Xtr, ytr)
        mask = selector.get_support()  # bool mask in original order
        mask_ordered = mask[order]

        # Color: selected vs not
        colors = np.where(mask_ordered, "#2E7D32", "#BDBDBD")  # green vs grey

        ax.barh(ordered_cols, ordered_scores, color=colors)
        ax.set_title(f"{name} (α={alpha})" if "SelectF" in name and name != "SelectKBest" else f"{name} (k={k})")
        ax.set_xlabel("ANOVA F score")
        # Legend proxy
        sel_patch = plt.Rectangle((0,0),1,1, color="#2E7D32", label="Selected")
        nsel_patch = plt.Rectangle((0,0),1,1, color="#BDBDBD", label="Not selected")
        ax.legend(handles=[sel_patch, nsel_patch], loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)



def plot_model_importance(model, X_cols, title: str):
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        labels = np.array(X_cols)
        order = np.argsort(vals)
        fig, ax = plt.subplots()
        ax.barh(labels[order], vals[order])
        ax.set_xlabel("Model importance")
        ax.set_title(title)
        st.pyplot(fig)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if isinstance(coef, (list, tuple)):
            coef = np.asarray(coef)
        coef = np.ravel(coef)
        vals = coef.astype(float)
        labels = np.array(X_cols)
        order = np.argsort(np.abs(vals))
        fig, ax = plt.subplots()
        ax.barh(labels[order], vals[order])
        ax.axvline(0, color='k', linewidth=1)
        ax.set_xlabel("Coefficient (signed)")
        ax.set_title(title)
        st.pyplot(fig)
    else:
        st.info("Model does not expose native importances/coefficients.")

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

# ------------------------------
# UI
# ------------------------------
# Beautified the app for publication

# Updated page configuration
st.set_page_config(page_title="Causal Mindset: Feature Selection", layout="wide", initial_sidebar_state="expanded")
st.title("🌟 Causal Mindset: Feature Selection Playground")
st.text("Adding features that provide unique information about the target variable can improve model performance.")
st.text("This becomes especially important when there is significant correlation among X2 - > X1 and X3 -> X1.")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.5em; font-weight: bold; color: #2E7D32; }
    .sidebar-header { font-size: 1.5em; font-weight: bold; color: #2E7D32; }
    .slider-label { font-size: 1.2em; font-weight: bold; }
    .dataframe { border: 1px solid #ddd; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar header
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Simulation Controls</div>", unsafe_allow_html=True)
    n = st.slider("n (samples)", 200, 5000, 1000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
    eps_sigma = st.slider("Noise σ (epsilon)", 0.05, 2.0, 0.5, step=0.05)
    model_type = st.selectbox("Model", ["RandomForest", "LinearRegression"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, step=0.05)

    st.divider()
    st.caption("y = β₁·X1 + β₄·X4 + ε, with X1 = a·X2 + b·X3 + c·ν")
    a = st.slider("a (X1 ← X2)", 0.0, 1.0, 0.75, 0.05)
    b = st.slider("b (X1 ← X3)", 0.0, 1.0, 0.70, 0.05)
    c = st.slider("c (X1 noise)", 0.0, 1.0, 0.20, 0.05)
    beta_x1 = st.slider("β₁ (y ← X1)", 0.0, 2.0, 1.15, 0.05)
    add_x4 = st.checkbox("Include X4 (direct)", value=True)
    beta_x4 = st.slider("β₄ (y ← X4)", 0.0, 2.0, 0.65, 0.05)
    non_linear_x1 = st.checkbox("Non-linear relationship: X1 → y", value=False)
    non_linear_x4 = st.checkbox("Non-linear relationship: X4 → y", value=False)
    default_feats = ["X1","X2","X3","X4"]
    include_features = st.multiselect("Features to feed model",
                                      options=["X1","X2","X3","X4"],
                                      default=default_feats)

# Main content
st.markdown("<div class='main-header'>Simulation Results</div>", unsafe_allow_html=True)
df = simulate_non_ts(n, seed, a, b, c, beta_x1, beta_x4, eps_sigma, add_x4, non_linear_x1, non_linear_x4)
st.subheader("Sample of Simulated Data")
st.dataframe(df.head(), use_container_width=True)
print_functional_form_non_ts(beta_x1, add_x4, beta_x4)

# Prepare data
y = df.pop("y").values
cols = [c for c in include_features if c in df.columns]
if not cols:
    st.warning("Please select at least one feature to train the model.")
    st.stop()
X = df[cols]

# Fit model
model, metrics, (Xtr, Xte, ytr, yte) = fit_model(X, y, model_type, test_size, seed)
plot_feature_selection_non_ts(Xtr, ytr, method_rfe=True)
plot_threshold_selectors_non_ts(Xtr, ytr, alpha=0.05, k=None)  # k defaults to ~half of features

st.subheader("Model Metrics")
st.write(pd.DataFrame([metrics]))

st.subheader("Correlation Heatmap")
df_corr = Xte.copy()
df_corr = df_corr.assign(y=yte)
plot_y_correlations(Xte, yte, "Feature–y Correlations (Test Set)")

st.subheader("Feature Importance")
plot_both_importances(model, Xte, yte, title_prefix=f"{model.__class__.__name__}")
