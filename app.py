import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

st.set_page_config(page_title="BiLSTM – Sequence Classifier", page_icon="↔️", layout="wide")

st.title("↔️ Bidirectional LSTM – Text Sequence Classifier")
st.markdown(
    "A **Bidirectional LSTM** reads sequences both forward and backward, capturing context from both directions. "
    "Trained here on **IMDB movie reviews** (binary sentiment classification)."
)

st.sidebar.header("⚙️ Settings")
vocab_size  = st.sidebar.slider("Vocab Size",      5000, 20000, 10000, step=5000)
max_len     = st.sidebar.slider("Max Sequence Length", 50, 500, 200, step=50)
embed_dim   = st.sidebar.slider("Embedding Dim",    32, 256, 64, step=32)
bilstm_u    = st.sidebar.slider("BiLSTM Units",     16, 128, 64, step=16)
epochs      = st.sidebar.slider("Epochs",            1,  10,  3)

st.subheader("📦 IMDB Dataset")
if st.button("Load IMDB & Train BiLSTM"):
    with st.spinner("Loading IMDB …"):
        (X_tr, y_tr), (X_te, y_te) = imdb.load_data(num_words=vocab_size)
        X_tr = pad_sequences(X_tr, maxlen=max_len, padding="post", truncating="post")
        X_te = pad_sequences(X_te, maxlen=max_len, padding="post", truncating="post")

    st.success(f"Train: {len(X_tr)} | Test: {len(X_te)}")
    st.write(f"Positive reviews: {y_tr.sum()} | Negative: {len(y_tr)-y_tr.sum()}")

    # label distribution
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Negative","Positive"], [(len(y_tr)-y_tr.sum()), y_tr.sum()],
           color=["#e74c3c","#2ecc71"])
    ax.set_title("Class Distribution"); st.pyplot(fig)

    with st.spinner("Building & training Bidirectional LSTM …"):
        model = keras.Sequential([
            layers.Embedding(vocab_size, embed_dim, input_length=max_len),
            layers.Bidirectional(layers.LSTM(bilstm_u, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(bilstm_u // 2)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_tr, y_tr, epochs=epochs,
                            batch_size=128, validation_split=0.1, verbose=0)

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    st.success(f"✅ Test Accuracy: **{acc*100:.2f}%** | Loss: **{loss:.4f}**")

    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["accuracy"],     label="Train")
        ax2.plot(history.history["val_accuracy"], label="Val")
        ax2.set_title("Accuracy"); ax2.legend(); st.pyplot(fig2)
    with col2:
        fig3, ax3 = plt.subplots()
        ax3.plot(history.history["loss"],     label="Train")
        ax3.plot(history.history["val_loss"], label="Val")
        ax3.set_title("Loss"); ax3.legend(); st.pyplot(fig3)

    st.session_state["bilstm_model"] = model
    word_idx = imdb.get_word_index()
    st.session_state["bilstm_word_idx"] = word_idx

# ── Live prediction ─────────────────────────────────────────────────────────
st.subheader("🔮 Predict a Custom Review")
review_text = st.text_area("Type a movie review:", "This film was absolutely fantastic!")
if st.button("Analyze Sentiment"):
    if "bilstm_model" not in st.session_state:
        st.warning("Train the model first.")
    else:
        word_idx = st.session_state["bilstm_word_idx"]
        tokens   = [word_idx.get(w.lower(), 2) + 3 for w in review_text.split()]
        tokens   = [min(t, vocab_size - 1) for t in tokens]
        padded   = pad_sequences([tokens], maxlen=max_len, padding="post", truncating="post")
        prob     = st.session_state["bilstm_model"].predict(padded, verbose=0)[0][0]
        sentiment = "😊 Positive" if prob > 0.5 else "😞 Negative"
        st.metric("Sentiment", sentiment, f"{max(prob, 1-prob)*100:.1f}% confidence")

st.markdown("---")
st.markdown("**Architecture:** Embedding → BiLSTM(fwd+bwd) → BiLSTM → Dense → Dropout → Sigmoid")
