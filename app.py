
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "handwritten_digit_cnn_best.keras"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Real-world robust preprocessing for handwritten digit images.

    Pipeline:
    grayscale -> normalization -> denoising -> adaptive threshold ->
    morphological noise removal -> bounding-box detection ->
    aspect-ratio preserving resize -> centering -> normalization
    """

    import cv2
    import numpy as np

    # Convert PIL image to NumPy
    img = np.array(image)

    # Handle RGB/RGBA input
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize image intensity
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Reduce camera/image noise
    blurred = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # Adaptive threshold for different lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )

    # Remove small isolated noise
    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Detect foreground
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    # Bounding box around the handwritten digit
    x, y, w, h = cv2.boundingRect(coords)

    # Add proportional padding
    padding = max(
        2,
        int(0.15 * max(w, h))
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(binary.shape[1], x + w + padding)
    y2 = min(binary.shape[0], y + h + padding)

    cropped = binary[y1:y2, x1:x2]

    # Preserve aspect ratio
    h_crop, w_crop = cropped.shape

    scale = 20 / max(
        h_crop,
        w_crop
    )

    new_w = max(
        1,
        int(w_crop * scale)
    )

    new_h = max(
        1,
        int(h_crop * scale)
    )

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # MNIST-style 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center digit
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Normalize
    processed = canvas.astype(
        np.float32
    ) / 255.0

    # CNN input
    return processed.reshape(
        1,
        28,
        28,
        1
    )

def predict_digit(processed_image):

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_digit]
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = [
        {
            "Digit": int(index),
            "Probability": float(
                probabilities[index]
            )
        }
        for index in top_indices
    ]

    return (
        predicted_digit,
        confidence,
        probabilities,
        top_predictions
    )


# ============================================================
# CONFIDENCE INTERPRETATION
# ============================================================

def confidence_message(confidence):

    if confidence >= 0.95:
        return (
            "🟢 Very High Confidence",
            "The model is highly confident in this prediction."
        )

    elif confidence >= 0.80:
        return (
            "🟡 High Confidence",
            "The model has a strong preference for this digit."
        )

    elif confidence >= 0.60:
        return (
            "🟠 Moderate Confidence",
            "The prediction is plausible, but some ambiguity exists."
        )

    else:
        return (
            "🔴 Low Confidence",
            "The handwriting may be ambiguous or different "
            "from the MNIST training distribution."
        )


# ============================================================
# SAVE PREDICTION TO HISTORY
# ============================================================

def save_prediction(
    predicted_digit,
    confidence
):

    st.session_state.prediction_history.insert(
        0,
        {
            "Prediction": predicted_digit,
            "Confidence": confidence * 100
        }
    )

    # Keep latest 10 predictions
    st.session_state.prediction_history = (
        st.session_state.prediction_history[:10]
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    processed_image,
    predicted_digit,
    confidence,
    probabilities,
    top_predictions
):

    st.divider()

    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    result_col, confidence_col = st.columns(2)

    with result_col:

        st.success(
            f"### Predicted Digit: **{predicted_digit}**"
        )

    with confidence_col:

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

    # --------------------------------------------------------
    # CONFIDENCE INTERPRETATION
    # --------------------------------------------------------

    message_title, message_text = (
        confidence_message(confidence)
    )

    st.info(
        f"**{message_title}**\n\n{message_text}"
    )

    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    st.subheader("🥇 Top-3 Predictions")

    top_cols = st.columns(3)

    for position, item in enumerate(
        top_predictions
    ):

        with top_cols[position]:

            st.metric(
                f"#{position + 1} — Digit {item['Digit']}",
                f"{item['Probability'] * 100:.2f}%"
            )

    # --------------------------------------------------------
    # TWO COLUMN ANALYSIS
    # --------------------------------------------------------

    analysis_col, image_col = st.columns(2)

    with analysis_col:

        st.subheader(
            "📊 Probability Distribution"
        )

        probability_df = pd.DataFrame({
            "Digit": np.arange(10),
            "Probability (%)": (
                probabilities * 100
            )
        })

        probability_df = probability_df.set_index(
            "Digit"
        )

        st.bar_chart(
            probability_df,
            height=350
        )

    with image_col:

        st.subheader(
            "🔍 What the CNN Sees"
        )

        st.image(
            processed_image[0].squeeze(),
            width=280,
            clamp=True
        )

        st.caption(
            "The input after grayscale conversion, "
            "cropping, centering, resizing and normalization."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔢 AI-Based Handwritten Digit Recognition System"
)

st.markdown(
    """
    ### 🧠 Deep Learning + Computer Vision

    Draw or upload a handwritten digit and let a trained
    Convolutional Neural Network recognize it.
    """
)

st.caption(
    "Built using TensorFlow/Keras and the MNIST dataset"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Model Information")

    st.write("**Architecture:** CNN")
    st.write("**Dataset:** MNIST")
    st.write("**Input:** 28 × 28 × 1")
    st.write("**Classes:** 10")
    st.write("**Test Accuracy:** 99.46%")
    st.write("**Best Validation:** 99.38%")

    st.divider()

    st.header("⚙️ Processing Pipeline")

    st.write("1. Input acquisition")
    st.write("2. Grayscale conversion")
    st.write("3. Background normalization")
    st.write("4. Digit cropping")
    st.write("5. Centering")
    st.write("6. 28×28 resizing")
    st.write("7. Pixel normalization")
    st.write("8. CNN prediction")

    st.divider()

    st.info(
        "The model was trained on normalized MNIST "
        "handwritten digit images."
    )


# ============================================================
# INPUT TABS
# ============================================================

draw_tab, upload_tab, analytics_tab, error_tab = st.tabs(
    [
        "✍️ Draw Digit",
        "📤 Upload Image",
        "📊 Model Analytics",
        "🔍 Error Analysis"
    ]
)


# ============================================================
# DRAW TAB
# ============================================================

with draw_tab:

    st.subheader(
        "✍️ Draw a Handwritten Digit"
    )

    st.write(
        "Draw one digit using your mouse or trackpad."
    )

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    if canvas_result.image_data is not None:

        canvas_array = (
            canvas_result.image_data
            .astype(np.uint8)
        )

        if np.max(canvas_array[:, :, 3]) > 0:

            canvas_image = Image.fromarray(
                canvas_array[:, :, :3]
            )

            if st.button(
                "🔍 Recognize Drawn Digit",
                type="primary",
                use_container_width=True
            ):

                processed_image = (
                    preprocess_image(canvas_image)
                )

                (
                    predicted_digit,
                    confidence,
                    probabilities,
                    top_predictions
                ) = predict_digit(
                    processed_image
                )

                save_prediction(
                    predicted_digit,
                    confidence
                )

                display_results(
                    processed_image,
                    predicted_digit,
                    confidence,
                    probabilities,
                    top_predictions
                )


# ============================================================
# MODEL ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.header("📊 Model Analytics Dashboard")

    st.markdown(
        """
        Explore the performance of the trained CNN model
        using the final MNIST test evaluation.
        """
    )

    # --------------------------------------------------------
    # OFFICIAL METRICS
    # --------------------------------------------------------

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            "Test Accuracy",
            "99.46%"
        )

    with metric_2:
        st.metric(
            "Test Loss",
            "0.016917"
        )

    with metric_3:
        st.metric(
            "Correct Predictions",
            "9,946 / 10,000"
        )

    with metric_4:
        st.metric(
            "Parameters",
            "944,490"
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📚 Dataset & Model Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write("**Dataset:** MNIST")
        st.write("**Training images:** 60,000")
        st.write("**Test images:** 10,000")
        st.write("**Classes:** 10")
        st.write("**Input shape:** 28 × 28 × 1")

    with summary_col2:

        st.write("**Architecture:** CNN")
        st.write("**Optimizer:** Adam")
        st.write("**Loss:** Sparse Categorical Crossentropy")
        st.write("**Test Accuracy:** 99.46%")
        st.write("**Error Rate:** 0.54%")

    st.divider()

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------

    st.subheader("📈 Training & Validation Performance")

    accuracy_curve = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "evaluation",
        "training_validation_accuracy.png"
    )

    loss_curve = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "evaluation",
        "training_validation_loss.png"
    )

    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:

        if os.path.exists(accuracy_curve):

            st.image(
                accuracy_curve,
                caption="Training vs Validation Accuracy",
                use_container_width=True
            )

        else:

            st.warning(
                "Accuracy curve not found."
            )

    with curve_col2:

        if os.path.exists(loss_curve):

            st.image(
                loss_curve,
                caption="Training vs Validation Loss",
                use_container_width=True
            )

        else:

            st.warning(
                "Loss curve not found."
            )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("🎯 Confusion Matrix")

    confusion_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "evaluation",
        "confusion_matrix.png"
    )

    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="Final Confusion Matrix — MNIST Test Set",
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "evaluation",
        "classification_report.txt"
    )

    if os.path.exists(report_path):

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as report_file:

            report_text = report_file.read()

        st.code(
            report_text,
            language="text"
        )

    else:

        st.warning(
            "Classification report not found."
        )


# ============================================================
# UPLOAD TAB
# ============================================================

with upload_tab:

    st.subheader(
        "📤 Upload a Handwritten Digit"
    )

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG or JPEG image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        original_image = Image.open(
            uploaded_file
        )

        st.subheader("Original Image")

        st.image(
            original_image,
            width=300
        )

        if st.button(
            "🔍 Recognize Uploaded Digit",
            type="primary",
            use_container_width=True
        ):

            processed_image = (
                preprocess_image(original_image)
            )

            (
                predicted_digit,
                confidence,
                probabilities,
                top_predictions
            ) = predict_digit(
                processed_image
            )

            save_prediction(
                predicted_digit,
                confidence
            )

            display_results(
                processed_image,
                predicted_digit,
                confidence,
                probabilities,
                top_predictions
            )


# ============================================================
# ERROR ANALYSIS TAB
# ============================================================

with error_tab:

    st.header("🔍 Model Error Analysis")

    st.markdown(
        """
        This section examines the cases where the final CNN
        model incorrectly classified MNIST test images.
        """
    )

    # --------------------------------------------------------
    # ERROR SUMMARY
    # --------------------------------------------------------

    error_col1, error_col2, error_col3 = st.columns(3)

    with error_col1:

        st.metric(
            "Test Images",
            "10,000"
        )

    with error_col2:

        st.metric(
            "Correct",
            "9,946"
        )

    with error_col3:

        st.metric(
            "Misclassified",
            "54"
        )

    st.info(
        "The final model has an error rate of 0.54%, "
        "with 54 misclassified images out of 10,000 test images."
    )

    st.divider()

    # --------------------------------------------------------
    # MOST COMMON ERROR PAIRS
    # --------------------------------------------------------

    st.subheader(
        "🔄 Most Frequent Confusion Pairs"
    )

    pairs_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "error_analysis",
        "error_confusion_pairs.csv"
    )

    if os.path.exists(pairs_path):

        pairs_df = pd.read_csv(
            pairs_path
        )

        st.dataframe(
            pairs_df.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Error confusion-pair data not found."
        )

    st.divider()

    # --------------------------------------------------------
    # VISUAL ERROR ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "🖼️ Misclassified Digit Examples"
    )

    error_grid_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "error_analysis",
        "misclassified_examples.png"
    )

    if os.path.exists(error_grid_path):

        st.image(
            error_grid_path,
            caption=(
                "Examples of incorrect predictions "
                "made by the final CNN model"
            ),
            use_container_width=True
        )

    else:

        st.warning(
            "Misclassified example image not found."
        )

    st.divider()

    # --------------------------------------------------------
    # ERROR INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "💡 Error Interpretation"
    )

    st.markdown(
        """
        The remaining errors are primarily associated with
        visually similar handwritten patterns.

        Examples include:

        - **6 → 0**
        - **2 → 7**
        - **9 → 4**
        - **5 → 3**
        - **4 → 9**

        These cases demonstrate that even a high-performing
        CNN can encounter ambiguity when handwritten shapes
        share similar visual characteristics.
        """
    )

    st.success(
        "The error-analysis module helps identify not only "
        "how accurate the model is, but also where its "
        "recognition performance can be improved."
    )



# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    st.divider()

    st.subheader(
        "🕘 Recent Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    history_df.index = (
        history_df.index + 1
    )

    history_df.index.name = "Attempt"

    history_df["Confidence"] = (
        history_df["Confidence"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Handwritten Digit Recognition System | "
    "Major Project | Deep Learning + Computer Vision"
)
