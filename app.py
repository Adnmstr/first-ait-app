import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    w = w.reshape(-1)  # ensure (d,)
    return X @ w + b

def mse_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE = (1/n) Σ (y - ŷ)^2
    """
    y = y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y - y_pred) ** 2))

def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    """
    Activity gradients:
      ∂L/∂w = -(2/n) X^T (y - ŷ)
      ∂L/∂b = -(2/n) Σ (y - ŷ)
    """
    n = X.shape[0]
    y = y.reshape(-1)
    y_pred = y_pred.reshape(-1)

    error = (y - y_pred)  # (n,)
    dw = -(2.0 / n) * (X.T @ error)  # (d,)
    db = -(2.0 / n) * np.sum(error)  # scalar

    return dw.reshape(-1), float(db)

def gradient_descent_train(X: np.ndarray, y: np.ndarray, lr: float, iters: int):
    """
    Train linear regression with gradient descent.
    X should already be standardized if you want fast/steady convergence.
    """
    n, d = X.shape
    rng = np.random.default_rng(0)

    # Initialize weights and bias
    w = rng.normal(0, 0.01, size=d)  # small random init
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
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

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

tabs = st.tabs(["Data", "Train", "Visualize", "Report Checklist"])

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

    # Feature scaling hint (recommended in activity)
    X_train_s, mean, std = standardize_fit_transform(X_train)
    X_test_s = standardize_transform(X_test, mean, std)

    if st.button("🚀 Train with Gradient Descent"):
        with st.spinner("Training..."):
            # Train from scratch on standardized X
            w, b, losses = gradient_descent_train(X_train_s, y_train, float(lr), int(iters))

            # Predict
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

        st.success("Done! Go to Visualize.")

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

        # Training progress plot (Loss vs Iteration)
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y=losses, mode="lines", name="MSE"))
        loss_fig.update_layout(title="Training Progress: Loss vs Iteration", xaxis_title="Iteration", yaxis_title="MSE")
        st.plotly_chart(loss_fig, use_container_width=True)

        # Predictions plot: Actual vs Predicted (test)
        y_test = preds["y_test"]
        y_test_pred = preds["y_test_pred"]
        pv_fig = px.scatter(x=y_test, y=y_test_pred, labels={"x": "Actual", "y": "Predicted"},
                            title="Predictions: Actual vs Predicted (Test Set)")
        st.plotly_chart(pv_fig, use_container_width=True)

        # Residuals plot
        residuals = y_test - y_test_pred
        res_fig = px.scatter(x=y_test_pred, y=residuals,
                             labels={"x": "Predicted", "y": "Residual (y - ŷ)"},
                             title="Residuals Plot")
        st.plotly_chart(res_fig, use_container_width=True)

# ---- Report checklist tab
with tabs[3]:
    st.subheader("Submission Checklist (quick)")
    st.write("- [ ] Data generation working")
    st.write("- [ ] Gradient descent implemented and converges")
    st.write("- [ ] Loss decreases over iterations")
    st.write("- [ ] Metrics shown: MSE, RMSE, MAE, R²")
    st.write("- [ ] Plots: loss curve, actual vs predicted, residuals")
    st.write("- [ ] Clean UI + sidebar controls")
    st.write("- [ ] Screenshots for submission")
    st.write("- [ ] 1–2 page brief report")
