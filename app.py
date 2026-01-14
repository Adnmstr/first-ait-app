# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Seaborn plots (extra "nice" visuals)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regression Analysis App", layout="wide")

# =========================
# Task 1: Data Generation
# =========================
@st.cache_data
def generate_synthetic_regression(n_samples: int, noise: float, seed: int):
    """
    Replace this with your course's provided Synthetic Dataset Generator call
    if you have it available in your project folder. The activity expects you
    to use the provided generator. (You can wire it in later.)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n_samples, 1))
    true_w, true_b = 2.0, 1.0
    y = true_w * X[:, 0] + true_b + rng.normal(0, noise, size=n_samples)
    return X, y, (true_w, true_b)

def standardize_fit_transform(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std

# =========================
# Task 2: Gradient Descent LR (FROM SCRATCH)
# =========================
def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    ŷ = Xw + b
    X: (n, d)
    w: (d,) or (d, 1)
    returns: (n,)
    """
    w = np.asarray(w).reshape(-1)  # ensure (d,)
    return X @ w + b

def mse_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE = (1/n) Σ (y - ŷ)^2
    """
    y = np.asarray(y).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y - y_pred) ** 2))

def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    """
    Activity gradients:
      ∂L/∂w = -(2/n) X^T (y - ŷ)
      ∂L/∂b = -(2/n) Σ (y - ŷ)
    """
    n = X.shape[0]
    y = np.asarray(y).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error = (y - y_pred)           # (n,)
    dw = -(2.0 / n) * (X.T @ error) # (d,)
    db = -(2.0 / n) * np.sum(error) # scalar
    return dw.reshape(-1), float(db)

def gradient_descent_train(X: np.ndarray, y: np.ndarray, lr: float, iters: int):
    """
    Train linear regression with gradient descent.
    X should already be standardized if you want fast/steady convergence.
    """
    _, d = X.shape
    rng = np.random.default_rng(0)

    # Initialize weights and bias
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    losses = []

    for _ in range(iters):
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        dw, db = compute_gradients(X, y, y_pred)

        # update params
        w = w - lr * dw
        b = b - lr * db

        losses.append(loss)

    return w, b, losses

# =========================
# Task 3: Metrics
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# =========================
# Seaborn Plot Helpers
# =========================
def seaborn_loss_plot(losses):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.lineplot(x=np.arange(len(losses)), y=np.array(losses), ax=ax)
    ax.set_title("Training Progress (Seaborn): Loss vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True, alpha=0.2)
    return fig

def seaborn_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.regplot(x=y_true, y=y_pred, ax=ax, scatter_kws={"alpha": 0.6})
    ax.set_title("Actual vs Predicted (Seaborn)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.2)
    return fig

def seaborn_residuals_plot(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residuals vs Predicted (Seaborn)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (y - ŷ)")
    ax.grid(True, alpha=0.2)
    return fig

def seaborn_residual_hist(residuals):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution (Seaborn)")
    ax.set_xlabel("Residual (y - ŷ)")
    ax.grid(True, alpha=0.2)
    return fig

# =========================
# Report generator
# =========================
def build_report_text(seed, n_samples, noise, test_split, lr, iters, metrics_train=None, metrics_test=None):
    def fmt(m, key):
        return "N/A" if m is None else f"{m[key]:.4f}"

    report = f"""
# Regression Analysis Brief Report (1–2 pages)

## Objective
This application implements **linear regression from scratch** using **gradient descent** and evaluates performance using standard regression metrics. The goal is to demonstrate the workflow: data generation, training, evaluation, and visualization.

## Data Generation
A synthetic linear dataset was generated with:
- **Samples (n):** {n_samples}
- **Noise (σ):** {noise}
- **Seed:** {seed}

True function (sanity check): **y ≈ 2.0x + 1.0** (before noise).

## Model and Training
Model form: **ŷ = Xw + b**

Loss: **MSE = (1/n) Σ (y − ŷ)²**

Gradients:
- **dw = -(2/n) Xᵀ(y − ŷ)**
- **db = -(2/n) Σ(y − ŷ)**

Settings:
- **Train/Test Split:** {test_split:.2f}
- **Learning rate (α):** {lr}
- **Iterations:** {iters}
- **Feature scaling:** X was standardized using train mean/std.

## Results
### Training Metrics
- **MSE:** {fmt(metrics_train, 'mse')}
- **RMSE:** {fmt(metrics_train, 'rmse')}
- **MAE:** {fmt(metrics_train, 'mae')}
- **R²:** {fmt(metrics_train, 'r2')}

### Test Metrics
- **MSE:** {fmt(metrics_test, 'mse')}
- **RMSE:** {fmt(metrics_test, 'rmse')}
- **MAE:** {fmt(metrics_test, 'mae')}
- **R²:** {fmt(metrics_test, 'r2')}

## Visual Checks
1. **Loss curve** should decrease if gradient descent converges.
2. **Actual vs Predicted** should align close to the diagonal for good fit.
3. **Residual plot** should show random scatter around zero.

## Conclusion
The model successfully trains via gradient descent and produces measurable predictive performance. Lower noise and appropriate hyperparameters generally improve accuracy and produce cleaner residual patterns.
""".strip()
    return report

# =========================
# UI
# =========================
st.title("📈 Regression Analysis App (Gradient Descent)")

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Seed", 0, 999999, 42, 1)
    n_samples = st.slider("Samples", 50, 2000, 300, 50)
    noise = st.slider("Noise", 0.0, 5.0, 1.0, 0.1)
    test_split = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)

    st.subheader("Gradient Descent")
    lr = st.number_input("Learning rate (α)", 0.0001, 1.0, 0.05, 0.01, format="%.4f")
    iters = st.slider("Iterations", 50, 5000, 800, 50)

tabs = st.tabs(["Data", "Train", "Visualize", "Report", "Checklist ✅"])

# ---- Data tab
with tabs[0]:
    st.subheader("Data Generation")
    X, y, (true_w, true_b) = generate_synthetic_regression(n_samples, noise, int(seed))

    df = pd.DataFrame({"x": X[:, 0], "y": y})
    st.dataframe(df.head(20), use_container_width=True)

    fig = px.scatter(df, x="x", y="y", title="Synthetic Regression Data")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"True function (for sanity check): y ≈ {true_w}x + {true_b}")

# ---- Train tab
with tabs[1]:
    st.subheader("Train Model")
    X, y, _ = generate_synthetic_regression(n_samples, noise, int(seed))

    # Split
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize
    X_train_s, mean, std = standardize_fit_transform(X_train)
    X_test_s = standardize_transform(X_test, mean, std)

    if st.button("🚀 Train with Gradient Descent"):
        with st.spinner("Training..."):
            w, b, losses = gradient_descent_train(X_train_s, y_train, float(lr), int(iters))

            y_train_pred = predict(X_train_s, w, b)
            y_test_pred = predict(X_test_s, w, b)

            metrics_train = compute_metrics(y_train, y_train_pred)
            metrics_test = compute_metrics(y_test, y_test_pred)

            st.session_state["model"] = {"w": w, "b": b, "mean": mean, "std": std}
            st.session_state["losses"] = losses
            st.session_state["preds"] = {
                "y_train": y_train, "y_train_pred": y_train_pred,
                "y_test": y_test, "y_test_pred": y_test_pred
            }
            st.session_state["metrics"] = {"train": metrics_train, "test": metrics_test}
            st.session_state["train_params"] = {
                "seed": int(seed),
                "n_samples": int(n_samples),
                "noise": float(noise),
                "test_split": float(test_split),
                "lr": float(lr),
                "iters": int(iters)
            }

        st.success("Done! Go to Visualize / Report / Checklist.")

# ---- Visualize tab
with tabs[2]:
    st.subheader("Visualizations")

    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        losses = st.session_state["losses"]
        preds = st.session_state["preds"]
        metrics = st.session_state["metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MSE", f"{metrics['train']['mse']:.3f}")
        c2.metric("Test MSE", f"{metrics['test']['mse']:.3f}")
        c3.metric("Test MAE", f"{metrics['test']['mae']:.3f}")
        c4.metric("Test R²", f"{metrics['test']['r2']:.3f}")

        st.markdown("### Plotly Charts")
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(losses) + 1)),
            y=losses,
            mode="lines",
            name="MSE"
        ))
        loss_fig.update_layout(
            title="Training Progress: Loss vs Iteration",
            xaxis_title="Iteration",
            yaxis_title="MSE"
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        y_test = preds["y_test"]
        y_test_pred = preds["y_test_pred"]
        pv_fig = px.scatter(
            x=y_test,
            y=y_test_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Predictions: Actual vs Predicted (Test Set)"
        )
        st.plotly_chart(pv_fig, use_container_width=True)

        residuals = y_test - y_test_pred
        res_fig = px.scatter(
            x=y_test_pred,
            y=residuals,
            labels={"x": "Predicted", "y": "Residual (y - ŷ)"},
            title="Residuals Plot"
        )
        st.plotly_chart(res_fig, use_container_width=True)

        st.markdown("### Seaborn Charts (Nice Looking)")
        colA, colB = st.columns(2)
        with colA:
            st.pyplot(seaborn_loss_plot(losses), clear_figure=True)
        with colB:
            st.pyplot(seaborn_actual_vs_pred(y_test, y_test_pred), clear_figure=True)

        colC, colD = st.columns(2)
        with colC:
            st.pyplot(seaborn_residuals_plot(y_test, y_test_pred), clear_figure=True)
        with colD:
            st.pyplot(seaborn_residual_hist(residuals), clear_figure=True)

# ---- Report tab
with tabs[3]:
    st.subheader("Brief Report (1–2 pages)")

    metrics_train = st.session_state.get("metrics", {}).get("train")
    metrics_test = st.session_state.get("metrics", {}).get("test")

    report_text = build_report_text(
        seed=int(seed),
        n_samples=int(n_samples),
        noise=float(noise),
        test_split=float(test_split),
        lr=float(lr),
        iters=int(iters),
        metrics_train=metrics_train,
        metrics_test=metrics_test
    )

    st.markdown(report_text)

    st.download_button(
        "Download Report (TXT)",
        data=report_text,
        file_name="regression_report.txt",
        mime="text/plain"
    )

# ---- Checklist tab
with tabs[4]:
    st.subheader("Submission Checklist")

    # Initialize checkbox state once
    if "checklist" not in st.session_state:
        st.session_state["checklist"] = {
            "data": False,
            "gd": False,
            "loss": False,
            "metrics": False,
            "plots": False,
            "ui": False,
            "screens": False,
            "report": False,
        }

    c = st.session_state["checklist"]

    c["data"] = st.checkbox("Data generation working", value=c["data"])
    c["gd"] = st.checkbox("Gradient descent implemented and converges", value=c["gd"])
    c["loss"] = st.checkbox("Loss decreases over iterations", value=c["loss"])
    c["metrics"] = st.checkbox("Metrics shown: MSE, RMSE, MAE, R²", value=c["metrics"])
    c["plots"] = st.checkbox("Plots: loss curve, actual vs predicted, residuals", value=c["plots"])
    c["ui"] = st.checkbox("Clean UI + sidebar controls", value=c["ui"])
    c["screens"] = st.checkbox("Screenshots for submission", value=c["screens"])
    c["report"] = st.checkbox("1–2 page brief report", value=c["report"])

    done_count = sum(bool(v) for v in c.values())
    total = len(c)

    st.divider()
    if done_count == total:
        st.success(f"All checklist items complete! ({done_count}/{total})")
    else:
        st.info(f"☑️ Progress: {done_count}/{total} complete")

