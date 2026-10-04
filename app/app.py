import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
import cv2
import json
import io

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "handwritten_digit_cnn_best.keras"
EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
FIG_DIR = BASE_DIR / "outputs" / "figures"

SUMMARY_PATH = EVAL_DIR / "cnn_evaluation_summary.json"
REPORT_PATH = EVAL_DIR / "cnn_classification_report.txt"

ACCURACY_PATH = FIG_DIR / "cnn_accuracy_curve.png"
LOSS_PATH = FIG_DIR / "cnn_loss_curve.png"
ARCHITECTURE_PATH = FIG_DIR / "cnn_architecture.png"
CONFUSION_PATH = EVAL_DIR / "cnn_confusion_matrix.png"
MISCLASSIFIED_PATH = EVAL_DIR / "cnn_misclassified_samples.png"
ERROR_PAIRS_PATH = EVAL_DIR / "cnn_error_pairs.png"
ERROR_PAIRS_CSV = EVAL_DIR / "cnn_error_pairs.csv"
MISCLASSIFIED_CSV = EVAL_DIR / "cnn_misclassified_samples.csv"

# ============================================================
# CUSTOM CSS — FINAL SUBMISSION LOOK
# ============================================================

st.markdown(
    """
<style>
.main-title {
    font-size: 40px;
    font-weight: 850;
    line-height: 1.15;
    margin-bottom: 4px;
}
.subtitle {
    font-size: 17px;
    opacity: 0.72;
    margin-bottom: 18px;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
    margin-bottom: 14px;
}
.hero-status {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    background: rgba(49,190,120,0.12);
    border: 1px solid rgba(49,190,120,0.30);
    color: #35c98a;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}
.prediction-card {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.035);
    text-align: center;
}
.prediction-digit {
    font-size: 86px;
    font-weight: 900;
    line-height: 1;
    margin: 8px 0 12px;
}
.confidence-badge {
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(49,190,120,0.10);
    border: 1px solid rgba(49,190,120,0.25);
    font-weight: 700;
}
.probability-title {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 10px;
}
.pipeline-item {
    padding: 5px 0;
    font-size: 14px;
}
.small-muted {
    opacity: 0.65;
    font-size: 13px;
}
.metric-note {
    opacity: 0.62;
    font-size: 12px;
}
div[data-testid="stMetric"] {
    padding: 4px 0;
}
div[data-testid="stTabs"] button {
    font-weight: 650;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# FAST CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    # compile=False avoids unnecessary optimizer restoration and
    # makes the application startup noticeably faster.
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data(show_spinner=False)
def load_summary():
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_text(path_string):
    path = Path(path_string)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_display_image(path_string, max_width=1250):
    """
    Load and downscale large visualization files once.
    This prevents Streamlit from repeatedly transferring huge PNGs.
    """
    path = Path(path_string)
    if not path.exists():
        return None

    image = Image.open(path).convert("RGB")
    if image.width > max_width:
        scale = max_width / image.width
        image = image.resize(
            (max_width, max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


model = load_model()
summary = load_summary()

TEST_ACCURACY = float(summary.get("test_accuracy", 0.996))
BEST_VALIDATION = float(summary.get("best_validation_accuracy", 0.9943333268))
TEST_LOSS = float(summary.get("test_loss", 0.013715486))
CORRECT = int(summary.get("correct_predictions", 9960))
INCORRECT = int(summary.get("incorrect_predictions", 40))
MEAN_CONFIDENCE = float(summary.get("mean_prediction_confidence", 0.997))
PER_DIGIT_ACCURACY = summary.get("per_digit_accuracy", {})

TOTAL_TEST = CORRECT + INCORRECT
ERROR_RATE = (INCORRECT / TOTAL_TEST * 100) if TOTAL_TEST else 0.0

# ============================================================
# ROBUST IMAGE PREPROCESSING
# ============================================================

def _prepare_rgb(image):
    """Convert input to RGB and limit very large uploads for speed."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image = image.convert("RGB")

    # Very large phone/camera images slow OpenCV and increase noise.
    max_side = 1400
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return image


def _background_reference(rgb):
    """Estimate background from the four image borders."""
    h, w = rgb.shape[:2]
    bw = max(2, int(min(h, w) * 0.04))

    strips = np.concatenate(
        [
            rgb[:bw].reshape(-1, 3),
            rgb[-bw:].reshape(-1, 3),
            rgb[:, :bw].reshape(-1, 3),
            rgb[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(strips, axis=0).astype(np.float32)


def _make_foreground_mask(rgb):
    """
    Color-aware foreground detection supporting:
    1) black digit on white/light background
    2) white digit on black/dark background
    3) colored digit on white/light background
    4) colored digit on dark background

    Returns a binary foreground mask and an estimated background polarity.
    """
    rgb_f = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    bg = _background_reference(rgb_f)
    bg_luma = float(cv2.cvtColor(
        np.uint8(np.clip(bg.reshape(1, 1, 3), 0, 255)),
        cv2.COLOR_RGB2GRAY
    )[0, 0])

    # Distance from the border background handles red/blue/green ink
    # even when grayscale intensity is weak.
    color_distance = np.linalg.norm(rgb_f - bg.reshape(1, 1, 3), axis=2)

    # Otsu threshold gives a fast global separation.
    otsu_value, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if bg_luma >= 128:
        # Light background: digit should be darker than the background.
        dark = gray < max(40, int(otsu_value))
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 7
        ) > 0
        color_mask = (saturation > 28) & (color_distance > 25)
        mask = dark | adaptive | color_mask
        polarity = "dark-on-light"
    else:
        # Dark background: digit should be lighter than the background.
        bright = gray > max(80, int(otsu_value))
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 7
        ) > 0
        color_mask = (saturation > 28) & (color_distance > 25)
        mask = bright | adaptive | color_mask
        polarity = "light-on-dark"

    # Always include strong color contrast.
    mask = mask | (color_distance > 45)

    # Remove tiny noise and connect broken handwriting strokes.
    mask_u8 = (mask.astype(np.uint8) * 255)

    # Ignore page/scan borders near the image edges. This is important
    # for photographed/scanned notebook or worksheet images.
    border_frac = 0.06
    bh = max(2, int(mask_u8.shape[0] * border_frac))
    bw = max(2, int(mask_u8.shape[1] * border_frac))
    mask_u8[:bh, :] = 0
    mask_u8[-bh:, :] = 0
    mask_u8[:, :bw] = 0
    mask_u8[:, -bw:] = 0

    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)

    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_OPEN, kernel_open, iterations=1
    )
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )

    # Remove tiny connected components while retaining meaningful strokes.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels > 1:
        image_area = mask_u8.shape[0] * mask_u8.shape[1]
        min_area = max(8, int(image_area * 0.00015))

        cleaned = np.zeros_like(mask_u8)
        components = []

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])

                cx = x + w / 2
                cy = y + h / 2
                image_cx = mask_u8.shape[1] / 2
                image_cy = mask_u8.shape[0] / 2
                center_distance = np.hypot(
                    (cx - image_cx) / max(1, mask_u8.shape[1]),
                    (cy - image_cy) / max(1, mask_u8.shape[0]),
                )

                score = area * (1.0 + max(0.0, 0.65 - center_distance))
                components.append((score, i))

        # Keep the strongest central/large components. This prevents
        # isolated background specks from expanding the crop.
        components.sort(reverse=True)

        # Prefer central, substantial components. For a single digit,
        # the largest central component is normally the handwriting.
        keep = [idx for _, idx in components[:4]]

        for idx in keep:
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])

            # Reject components that touch the image boundary; these are
            # commonly page borders, scan lines, or UI artifacts.
            touches_edge = (
                x <= 1 or y <= 1 or
                x + w >= mask_u8.shape[1] - 1 or
                y + h >= mask_u8.shape[0] - 1
            )
            if not touches_edge:
                cleaned[labels == idx] = 255

        # If all useful components were rejected, retain the strongest one.
        if cv2.countNonZero(cleaned) == 0 and components:
            cleaned[labels == components[0][1]] = 255

        mask_u8 = cleaned

    return mask_u8, polarity, bg


def preprocess_digit(image):
    """
    Convert an arbitrary single-digit image into MNIST-like 28x28 input.

    Returns:
        normalized_model_input
        display_image
        diagnostics
    """
    image = _prepare_rgb(image)
    rgb = np.array(image)

    mask, polarity, bg = _make_foreground_mask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    coords = cv2.findNonZero(mask)

    if coords is None:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank, np.zeros((28, 28), dtype=np.uint8), {
            "status": "No foreground detected",
            "polarity": polarity,
            "bounding_box": "None",
            "scale": 0.0,
        }

    x, y, w, h = cv2.boundingRect(coords)

    # Safety: reject a crop that is essentially the whole image.
    image_h, image_w = mask.shape
    crop_area_ratio = (w * h) / float(image_h * image_w)

    if crop_area_ratio > 0.96:
        # Try a stricter mask using background distance.
        bg_distance = np.linalg.norm(
            rgb.astype(np.float32) - bg.reshape(1, 1, 3),
            axis=2,
        )
        strict = (bg_distance > 35).astype(np.uint8) * 255
        strict = cv2.morphologyEx(
            strict, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        strict_coords = cv2.findNonZero(strict)

        if strict_coords is not None:
            sx, sy, sw, sh = cv2.boundingRect(strict_coords)
            if (sw * sh) < (w * h):
                mask = strict
                coords = strict_coords
                x, y, w, h = sx, sy, sw, sh

    # Add a small proportional padding around the detected digit.
    pad = max(2, int(round(0.10 * max(w, h))))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_w, x + w + pad)
    y1 = min(image_h, y + h + pad)

    crop_mask = mask[y0:y1, x0:x1]
    crop_gray = gray[y0:y1, x0:x1]

    # Build a brightness image where the foreground is always bright.
    # This makes black, white and colored writing compatible with MNIST.
    if polarity == "light-on-dark":
        ink = crop_gray
    else:
        ink = 255 - crop_gray

    # Color-aware strokes can be dark in grayscale. Use foreground mask
    # to recover strong ink intensity without bringing the background in.
    ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX)
    ink = cv2.bitwise_and(ink, crop_mask)

    # Slight smoothing gives more MNIST-like anti-aliased edges.
    ink = cv2.GaussianBlur(ink, (3, 3), 0)

    # Preserve aspect ratio with a 20x20 maximum, matching common MNIST
    # digit geometry while retaining the original proportions.
    ch, cw = ink.shape
    max_dim = max(cw, ch)

    scale = 20.0 / max_dim
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    digit = cv2.resize(
        ink,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center by image geometry first.
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit

    # Fine center using the foreground centroid.
    ys, xs = np.where(canvas > max(10, int(canvas.max() * 0.10)))

    if len(xs) > 10:
        cx = float(xs.mean())
        cy = float(ys.mean())

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        transform = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y],
        ])

        canvas = cv2.warpAffine(
            canvas,
            transform,
            (28, 28),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    normalized = canvas.astype(np.float32) / 255.0

    diagnostics = {
        "status": "Foreground detected",
        "polarity": polarity,
        "bounding_box": f"{w} × {h}px",
        "crop_area": f"{crop_area_ratio * 100:.1f}%",
        "scale": f"{scale:.4f}",
    }

    return normalized, canvas, diagnostics


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(image):
    processed, display_image, diagnostics = preprocess_digit(image)

    model_input = processed.reshape(1, 28, 28, 1)

    probabilities = model.predict(
        model_input,
        verbose=0,
    )[0]

    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    return (
        prediction,
        confidence,
        probabilities,
        processed,
        display_image,
        diagnostics,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def confidence_label(confidence):
    if confidence >= 0.95:
        return "Very High Confidence"
    if confidence >= 0.80:
        return "High Confidence"
    if confidence >= 0.60:
        return "Moderate Confidence"
    return "Low Confidence"


def display_probability_distribution(probabilities, prediction):
    st.markdown(
        '<div class="probability-title">📊 Prediction Probability Distribution</div>',
        unsafe_allow_html=True,
    )

    for digit, probability in enumerate(probabilities):
        probability = float(probability)
        percentage = probability * 100

        if digit == prediction:
            st.markdown(
                f"""
                <div class="confidence-badge">
                    🎯 Digit {digit} — {percentage:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="small-muted">Digit {digit} — {percentage:.2f}%</div>',
                unsafe_allow_html=True,
            )

        st.progress(min(max(probability, 0.0), 1.0))


def display_prediction_result(
    prediction,
    confidence,
    probabilities,
    display_image,
    diagnostics,
):
    result_left, result_right = st.columns([1, 1.65])

    with result_left:
        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True,
        )
        st.write("### 🧠 Predicted Digit")
        st.markdown(
            f'<div class="prediction-digit">{prediction}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="confidence-badge">{confidence * 100:.2f}% · {confidence_label(confidence)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:
        display_probability_distribution(probabilities, prediction)

    st.divider()

    st.markdown("### 🔬 What the CNN Sees")

    preview_col, info_col = st.columns([1, 2])

    with preview_col:
        st.image(
            display_image,
            width=260,
            caption="Final 28 × 28 model input",
        )

    with info_col:
        st.markdown("#### Processing Diagnostics")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Detection", diagnostics["status"])
        with d2:
            st.metric("Polarity", diagnostics["polarity"])
        with d3:
            st.metric("Crop", diagnostics["bounding_box"])

        st.caption(
            "The input is background-normalized, foreground-isolated, "
            "aspect-ratio preserved, centered and resized to 28 × 28."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 Model Information")
    st.markdown("**Architecture:** CNN")
    st.markdown("**Dataset:** MNIST")
    st.markdown("**Input:** 28 × 28 × 1")
    st.markdown("**Classes:** 10")
    st.markdown(f"**Test Accuracy:** {TEST_ACCURACY * 100:.2f}%")
    st.markdown(f"**Best Validation:** {BEST_VALIDATION * 100:.2f}%")

    st.divider()

    st.markdown("## ⚙️ Processing Pipeline")

    pipeline = [
        "1. Input acquisition",
        "2. RGB/color-aware foreground detection",
        "3. Background normalization",
        "4. Noise reduction",
        "5. Digit extraction",
        "6. Aspect-ratio preservation",
        "7. Centering",
        "8. 28 × 28 conversion",
        "9. CNN prediction",
    ]

    for item in pipeline:
        st.markdown(
            f'<div class="pipeline-item">{item}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("## 📊 Final Performance")
    st.metric("Test Accuracy", f"{TEST_ACCURACY * 100:.2f}%")
    st.metric("Correct", f"{CORRECT:,}")
    st.metric("Errors", f"{INCORRECT:,}")
    st.metric("Test Loss", f"{TEST_LOSS:.5f}")
    st.metric("Mean Confidence", f"{MEAN_CONFIDENCE * 100:.2f}%")

    st.divider()
    st.caption("CNN • TensorFlow/Keras • MNIST • OpenCV • Streamlit")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="hero-status">● SYSTEM READY · CNN MODEL LOADED</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">✍️ AI-Based Handwritten Digit Recognition System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Deep Learning based handwritten digit classification using a Convolutional Neural Network</div>',
    unsafe_allow_html=True,
)

# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🎯 Test Accuracy", f"{TEST_ACCURACY * 100:.2f}%")
with c2:
    st.metric("📈 Validation Accuracy", f"{BEST_VALIDATION * 100:.2f}%")
with c3:
    st.metric("✅ Correct", f"{CORRECT:,}")
with c4:
    st.metric("❌ Errors", f"{INCORRECT:,}")

st.divider()

# ============================================================
# MAIN TABS — SIMULTANEOUS ORIGINAL LAYOUT
# ============================================================

draw_tab, upload_tab, analytics_tab, error_tab = st.tabs(
    [
        "✍️ Draw Digit",
        "📤 Upload Image",
        "📊 Model Analytics",
        "🔍 Error Analysis",
    ]
)

# ============================================================
# DRAW DIGIT TAB
# ============================================================

with draw_tab:
    st.markdown(
        '<div class="section-title">✍️ Draw a Handwritten Digit</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Draw one digit from 0–9. The system automatically removes the "
        "black canvas background, isolates the stroke and converts it to "
        "the MNIST-style 28 × 28 model input."
    )

    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="digit_canvas",
            update_streamlit=True,
            return_image_data=True,
        )

        if st.button(
            "🔮 Recognize Drawn Digit",
            type="primary",
            use_container_width=True,
            key="recognize_drawn",
        ):
            if canvas_result.image_data is None:
                st.warning("Please draw a digit first.")
            else:
                canvas_image = Image.fromarray(
                    canvas_result.image_data.astype(np.uint8)
                )

                (
                    prediction,
                    confidence,
                    probabilities,
                    processed,
                    display_image,
                    diagnostics,
                ) = predict_digit(canvas_image)

                display_prediction_result(
                    prediction,
                    confidence,
                    probabilities,
                    display_image,
                    diagnostics,
                )

    except ImportError:
        st.error(
            "streamlit-drawable-canvas is not installed. "
            "Install it with: pip install streamlit-drawable-canvas"
        )

# ============================================================
# UPLOAD IMAGE TAB
# ============================================================

with upload_tab:
    st.markdown(
        '<div class="section-title">📤 Upload a Handwritten Digit</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload one handwritten digit as PNG, JPG, JPEG or BMP. "
        "Black, white and colored handwriting are supported. "
        "For best accuracy, use a clear single digit with good contrast and "
        "avoid page borders or multiple digits."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="digit_uploader",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            original_col, action_col = st.columns([1, 1.25])

            with original_col:
                st.markdown("### 🖼️ Original Image")
                st.image(image, width=360)

            with action_col:
                st.markdown("### 🚀 Recognition")
                st.caption(
                    f"Image: {image.width} × {image.height}px · "
                    f"{uploaded_file.type}"
                )

                if st.button(
                    "🔮 Recognize Uploaded Digit",
                    type="primary",
                    use_container_width=True,
                    key="recognize_uploaded",
                ):
                    (
                        prediction,
                        confidence,
                        probabilities,
                        processed,
                        display_image,
                        diagnostics,
                    ) = predict_digit(image)

                    display_prediction_result(
                        prediction,
                        confidence,
                        probabilities,
                        display_image,
                        diagnostics,
                    )

        except Exception as error:
            st.error(f"Unable to process this image: {error}")

# ============================================================
# MODEL ANALYTICS TAB
# ============================================================

with analytics_tab:
    st.markdown(
        '<div class="section-title">📊 Model Analytics</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Comprehensive evaluation of the final CNN on the 10,000-image MNIST test dataset."
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric("🎯 Test Accuracy", f"{TEST_ACCURACY * 100:.2f}%")
    with a2:
        st.metric("📉 Test Loss", f"{TEST_LOSS:.5f}")
    with a3:
        st.metric("✅ Correct", f"{CORRECT:,}")
    with a4:
        st.metric("❌ Error Rate", f"{ERROR_RATE:.2f}%")

    st.divider()

    # Training curves
    st.markdown("### 📈 Training & Validation Performance")
    curve_left, curve_right = st.columns(2)

    accuracy_img = load_display_image(str(ACCURACY_PATH), 1100)
    loss_img = load_display_image(str(LOSS_PATH), 1100)

    with curve_left:
        if accuracy_img is not None:
            st.image(accuracy_img, caption="CNN Accuracy Curve")
        else:
            st.warning("Accuracy curve not found.")

    with curve_right:
        if loss_img is not None:
            st.image(loss_img, caption="CNN Loss Curve")
        else:
            st.warning("Loss curve not found.")

    st.divider()

    # Confusion matrix and architecture
    st.markdown("### 🧩 Evaluation & Architecture")
    cm_col, arch_col = st.columns(2)

    confusion_img = load_display_image(str(CONFUSION_PATH), 1050)
    architecture_img = load_display_image(str(ARCHITECTURE_PATH), 1050)

    with cm_col:
        st.markdown("#### Confusion Matrix")
        if confusion_img is not None:
            st.image(confusion_img, caption="CNN Confusion Matrix")
        else:
            st.warning("Confusion matrix not found.")

    with arch_col:
        st.markdown("#### CNN Architecture")
        if architecture_img is not None:
            st.image(architecture_img, width=850, caption="Final CNN Architecture")
        else:
            st.warning("CNN architecture image not found.")

    st.divider()

    # Per-digit accuracy — clearly visible data
    st.markdown("### 🔢 Per-Digit Recognition Accuracy")

    if PER_DIGIT_ACCURACY:
        digit_rows = []
        for digit in range(10):
            value = float(PER_DIGIT_ACCURACY.get(str(digit), 0.0))
            digit_rows.append(
                {
                    "Digit": digit,
                    "Accuracy": f"{value * 100:.2f}%",
                    "Correctness": value,
                }
            )

        digit_df = pd.DataFrame(digit_rows)
        display_df = digit_df[["Digit", "Accuracy"]]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        chart_df = pd.DataFrame(
            {
                "Digit": [str(i) for i in range(10)],
                "Accuracy (%)": [
                    float(PER_DIGIT_ACCURACY.get(str(i), 0.0)) * 100
                    for i in range(10)
                ],
            }
        ).set_index("Digit")

        st.bar_chart(chart_df, use_container_width=True)

    st.divider()

    # Classification report
    st.markdown("### 📋 Classification Report")
    report_text = load_text(str(REPORT_PATH))

    if report_text:
        st.code(report_text, language="text")
    else:
        st.info("Classification report file not found.")

    st.divider()

    st.markdown("### 🧠 Final Model Summary")

    model_summary = pd.DataFrame(
        [
            ["Model", "Convolutional Neural Network"],
            ["Dataset", "MNIST"],
            ["Input", "28 × 28 × 1"],
            ["Classes", "10"],
            ["Test Samples", f"{TOTAL_TEST:,}"],
            ["Test Accuracy", f"{TEST_ACCURACY * 100:.2f}%"],
            ["Best Validation Accuracy", f"{BEST_VALIDATION * 100:.2f}%"],
            ["Test Loss", f"{TEST_LOSS:.5f}"],
            ["Correct Predictions", f"{CORRECT:,}"],
            ["Incorrect Predictions", f"{INCORRECT:,}"],
            ["Mean Prediction Confidence", f"{MEAN_CONFIDENCE * 100:.2f}%"],
        ],
        columns=["Metric", "Value"],
    )

    st.dataframe(
        model_summary,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# ERROR ANALYSIS TAB
# ============================================================

with error_tab:
    st.markdown(
        '<div class="section-title">🔍 Error Analysis</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Analysis of the 40 incorrect predictions made by the final CNN."
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric("Total Test Samples", f"{TOTAL_TEST:,}")
    with e2:
        st.metric("Correct Predictions", f"{CORRECT:,}")
    with e3:
        st.metric("Error Rate", f"{ERROR_RATE:.2f}%")

    st.divider()

    misclassified_img = load_display_image(str(MISCLASSIFIED_PATH), 1200)
    error_pairs_img = load_display_image(str(ERROR_PAIRS_PATH), 1200)

    st.markdown("### ❌ Misclassified Samples")
    if misclassified_img is not None:
        st.image(misclassified_img, caption="Examples of incorrect CNN predictions")
    else:
        st.info("Misclassified sample image not found.")

    st.divider()

    st.markdown("### 🔄 Common Error Pairs")
    if error_pairs_img is not None:
        st.image(error_pairs_img, caption="Most frequent true-label → predicted-label errors")
    else:
        st.info("Error-pair visualization not found.")

    st.divider()

    if ERROR_PAIRS_CSV.exists():
        st.markdown("### 📊 Error Pair Statistics")
        error_df = load_csv(str(ERROR_PAIRS_CSV))
        if not error_df.empty:
            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True,
            )

    if MISCLASSIFIED_CSV.exists():
        with st.expander("📄 View Misclassified Sample Records"):
            misclassified_df = load_csv(str(MISCLASSIFIED_CSV))
            if not misclassified_df.empty:
                st.dataframe(
                    misclassified_df,
                    use_container_width=True,
                    hide_index=True,
                )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
        <b>AI-Based Handwritten Digit Recognition System</b><br>
        CNN • TensorFlow/Keras • MNIST • OpenCV • Streamlit
        <br><br>
        Final CNN Test Accuracy: <b>99.60%</b>
        &nbsp;•&nbsp;
        9,960 / 10,000 correct predictions
    </center>
    """,
    unsafe_allow_html=True,
)
