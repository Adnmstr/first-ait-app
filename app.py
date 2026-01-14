"""
Streamlit App: Linear Regression from Scratch (Gradient Descent)
Run:
    streamlit run app.py

requirements.txt:
    streamlit
    pandas
    numpy
    plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="🎮 Linear Regression from Scratch (GD)",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Problem setup
# =========================
FEATURES = ["sessions_week", "levels_week", "days_since_last_play", "purchases_week"]
TARGET = "minutes_next_week"


# =========================
# Utilities
# =========================
@st.cache_data
def make_synthetic_game_data(n: int, seed: int) -> pd.DataFrame:
    """
    Synthetic game telemetry with a linear-ish relationship + noise.
    """
    rng = np.random.default_rng(seed)

    sessions = np.clip(rng.normal(9, 4, n), 0, None)
    levels = np.clip(rng.normal(14, 8, n), 0, None)
    days_since = np.clip(rng.normal(2.5, 2.5, n), 0, None)
    purchases = np.clip(rng.poisson(0.6, n), 0, None)

    # "True" underlying linear signal (unknown to the learner)
    # More sessions/levels -> more minutes next week
    # More days since last play -> fewer minutes next week
    # Purchases -> slightly more minutes next week
    noise = rng.normal(0, 35, n)

    minutes_next = (
        40
        + 18 * sessions
        + 6 * levels
        - 35 * days_since
        + 25 * purchases
        + noise
    )
    minutes_next = np.clip(minutes_next, 0, None)

    return pd.DataFrame(
        {
            "sessions_week": sessions.round(2),
            "levels_week": levels.round(2),
            "days_since_last_play": days_since.round(2),
            "purchases_week": purchases.astype(int),
            "minutes_next_week": minutes_next.round(2),
        }
    )


def train_test_split_np(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit_transform(X: np.ndarray):
    """
    Standardize features: (X - mean) / std
    Returns X_std, mean, std
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero
    Xs = (X - mean) / std
    return Xs, mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def gradient_descent_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
):
    """
    Batch Gradient Descent for linear regression:
        y_hat = X @ w + b

    Gradient:
        dw = (2/n) X^T (y_hat - y) + 2*l2*w
        db = (2/n) sum(y_hat - y)

    Returns:
        w, b, history (list of mse per epoch)
    """
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    history = []
    for _ in range(epochs):
        y_hat = X @ w + b
        error = y_hat - y

        dw = (2.0 / n) * (X.T @ error) + 2.0 * l2 * w
        db = (2.0 / n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        history.append(mse(y, X @ w + b))

    return w, b, history


# =========================
# Session State
# =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None  # dict storing w,b,mean,std
if "train_info" not in st.session_state:
    st.session_state.train_info = None  # metrics + history


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Controls")

data_mode = st.sidebar.radio("Data source", ["Synthetic (recommended)", "Upload CSV"])
seed = st.sidebar.number_input("Seed", 0, 999999, 42, 1)
n_rows = st.sidebar.slider("Synthetic rows", 200, 5000, 1200, 100)

st.sidebar.divider()
st.sidebar.subheader("🧠 Gradient Descent Settings")
lr = st.sidebar.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.05, step=0.01, format="%.5f")
epochs = st.sidebar.slider("Epochs", 50, 3000, 600, 50)
l2 = st.sidebar.number_input("L2 regularization (lambda)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.3f")
test_size = st.sidebar.slider("Test split", 0.1, 0.5, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.caption("This app uses pure NumPy for training (no sklearn).")


# =========================
# Main UI
# =========================
st.title("🎮 Linear Regression from Scratch — Gradient Descent")
st.markdown(
    """
**Goal:** predict **minutes played next week** from simple game telemetry using **batch gradient descent**.
- We standardize features (mean/std)
- Train `w` and `b` with gradient descent
- Track loss (MSE) over epochs
"""
)

tabs = st.tabs(["📦 Data", "🧠 Train", "📊 Results", "🕹️ Predict"])


# =========================
# Tab: Data
# =========================
with tabs[0]:
    st.subheader("📦 Dataset")

    if data_mode == "Synthetic (recommended)":
        df = make_synthetic_game_data(n_rows, seed)
        st.session_state.df = df
        st.success(f"Generated synthetic dataset with {len(df):,} rows.")
    else:
        st.info("Upload a CSV with these columns:")
        st.code("\n".join([*FEATURES, TARGET]), language="text")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                missing = [c for c in FEATURES + [TARGET] if c not in df_up.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    # basic numeric check
                    bad = [c for c in FEATURES + [TARGET] if not pd.api.types.is_numeric_dtype(df_up[c])]
                    if bad:
                        st.error(f"These columns must be numeric: {bad}")
                    else:
                        st.session_state.df = df_up.copy()
                        st.success(f"Loaded {len(df_up):,} rows.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    df = st.session_state.df
    if df is None:
        st.warning("No data yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", f"{len(FEATURES)}")
        c3.metric("Target mean", f"{df[TARGET].mean():.1f} min")

        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### Quick view")
        fig = px.scatter(
            df,
            x="sessions_week",
            y=TARGET,
            opacity=0.5,
            title="Minutes Next Week vs Sessions/Week",
            trendline=None,
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Tab: Train
# =========================
with tabs[1]:
    st.subheader("🧠 Train with Gradient Descent")

    df = st.session_state.df
    if df is None:
        st.warning("Load data first (Data tab).")
    else:
        st.markdown(
            """
**Model:**  
\\[
\\hat{y} = Xw + b
\\]
**Loss:** Mean Squared Error (MSE)  
Training uses **batch gradient descent** over all samples each epoch.
"""
        )

        if st.button("🚀 Train now"):
            X = df[FEATURES].to_numpy(dtype=float)
            y = df[TARGET].to_numpy(dtype=float)

            X_train, X_test, y_train, y_test = train_test_split_np(X, y, float(test_size), int(seed))
            X_train_s, mean, std = standardize_fit_transform(X_train)
            X_test_s = standardize_transform(X_test, mean, std)

            with st.spinner("Training (NumPy GD)…"):
                w, b, history = gradient_descent_linear_regression(
                    X_train_s, y_train, lr=float(lr), epochs=int(epochs), l2=float(l2)
                )

            # Evaluate
            y_train_pred = X_train_s @ w + b
            y_test_pred = X_test_s @ w + b

            train_mse = mse(y_train, y_train_pred)
            test_mse = mse(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            st.session_state.model = {"w": w, "b": b, "mean": mean, "std": std}
            st.session_state.train_info = {
                "history": history,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "X_test_s": X_test_s,
                "y_test": y_test,
                "y_test_pred": y_test_pred,
            }

            st.success("Done! Go to Results and Predict tabs.")


# =========================
# Tab: Results
# =========================
with tabs[2]:
    st.subheader("📊 Results")

    info = st.session_state.train_info
    model = st.session_state.model

    if info is None or model is None:
        st.info("Train the model first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MSE", f"{info['train_mse']:.1f}")
        c2.metric("Test MSE", f"{info['test_mse']:.1f}")
        c3.metric("Train R²", f"{info['train_r2']:.3f}")
        c4.metric("Test R²", f"{info['test_r2']:.3f}")

        st.markdown("### Learned parameters")
        w = model["w"]
        b = model["b"]

        param_df = pd.DataFrame({"feature": FEATURES, "weight (w)": w})
        st.dataframe(param_df, use_container_width=True)
        st.write(f"**bias (b):** {b:.3f}")

        st.markdown("### Loss curve (MSE vs epoch)")
        hist = info["history"]
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=list(range(1, len(hist) + 1)), y=hist, mode="lines", name="MSE"))
        loss_fig.update_layout(xaxis_title="Epoch", yaxis_title="MSE", height=380)
        st.plotly_chart(loss_fig, use_container_width=True)

        st.markdown("### Predicted vs actual (test set)")
        y_test = info["y_test"]
        y_pred = info["y_test_pred"]

        pv_fig = px.scatter(
            x=y_test,
            y=y_pred,
            opacity=0.55,
            labels={"x": "Actual minutes next week", "y": "Predicted minutes next week"},
            title="Test set: Actual vs Predicted",
        )
        # y=x reference line
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        pv_fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="y = x"))
        st.plotly_chart(pv_fig, use_container_width=True)


# =========================
# Tab: Predict
# =========================
with tabs[3]:
    st.subheader("🕹️ Predict a player's minutes next week")

    model = st.session_state.model
    if model is None:
        st.info("Train the model first.")
    else:
        st.markdown("Use sliders to create a hypothetical player and predict next-week minutes.")

        left, right = st.columns([1.2, 1])

        with left:
            sessions_week = st.slider("Sessions this week", 0, 40, 10, 1)
            levels_week = st.slider("Levels completed this week", 0, 80, 15, 1)
            days_since_last_play = st.slider("Days since last play", 0, 30, 2, 1)
            purchases_week = st.slider("Purchases this week", 0, 10, 0, 1)

            x = np.array([[sessions_week, levels_week, days_since_last_play, purchases_week]], dtype=float)
            x_s = standardize_transform(x, model["mean"], model["std"])
            y_hat = float(x_s @ model["w"] + model["b"])

        with right:
            st.metric("Predicted minutes next week", f"{max(0.0, y_hat):.1f} min")

            st.markdown("**Interpretation idea (game dev angle):**")
            st.write("- More sessions/levels generally increase predicted playtime.")
            st.write("- A larger 'days since last play' pushes the prediction down (possible churn risk).")
            st.write("- Purchases can correlate with engagement (depending on your game).")

        st.divider()
        st.caption("We standardized features so gradient descent behaves better.")