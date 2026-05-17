import gradio as gr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# =========================================================
# IMAGE ENHANCEMENT ALGORITHMS
# =========================================================

# 1. Histogram Enhancement

def histogram_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return enhanced


# 2. Gamma Correction

def gamma_correction(image, gamma=2.0):
    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


# 3. ML Enhancement

def deep_enhancement(image):
    img_float = image / 255.0

    enhanced = np.power(img_float, 0.6)

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    enhanced = cv2.fastNlMeansDenoisingColored(
        (enhanced * 255).astype(np.uint8),
        None,
        10,
        10,
        7,
        21
    )

    return enhanced


# 4. Log Transformation

def log_transformation(image):
    img_float = image / 255.0

    log_img = np.log1p(img_float)

    log_img = cv2.normalize(
        log_img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return log_img.astype(np.uint8)


# 5. Bilateral Filter

def bilateral_enhancement(image):
    return cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )


# 6. Sharpen + Contrast

def sharpen_contrast(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(
        image,
        -1,
        kernel
    )

    sharp = cv2.normalize(
        sharp,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return sharp


# 7. Zero-DCE

def zero_dce_enhancement(image):
    img = image / 255.0

    for _ in range(5):
        img = img + 0.5 * (img - img * img)

    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


# 8. Retinex Enhancement

def retinex_enhancement(image):
    img = image.astype(np.float32) + 1.0

    retinex = np.log10(img) - np.log10(
        cv2.GaussianBlur(
            img,
            (15, 15),
            0
        )
    )

    retinex = cv2.normalize(
        retinex,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(retinex)


# 9. CNN Style Enhancement

def cnn_style_enhancement(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    enhanced = cv2.filter2D(
        image,
        -1,
        kernel
    )

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        10,
        10,
        7,
        21
    )

    return enhanced


# =========================================================
# METRICS
# =========================================================

def mse(original, enhanced):
    return np.mean((original - enhanced) ** 2)



def psnr(original, enhanced):
    mse_value = mse(original, enhanced)

    if mse_value == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(mse_value))



def calculate_ssim(original, enhanced):
    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2GRAY
    )

    return ssim(original_gray, enhanced_gray)



def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)



def contrast_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.std(gray)



def sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# =========================================================
# MAIN PROCESS FUNCTION
# =========================================================

def process_image(input_image):

    img = input_image.copy()

    # Apply all algorithms
    hist_img = histogram_enhancement(img)
    gamma_img = gamma_correction(img)
    ml_img = deep_enhancement(img)
    log_img = log_transformation(img)
    bilateral_img = bilateral_enhancement(img)
    sharp_img = sharpen_contrast(img)
    zero_img = zero_dce_enhancement(img)
    retinex_img = retinex_enhancement(img)
    cnn_img = cnn_style_enhancement(img)

    algorithms = {
        "Histogram": hist_img,
        "Gamma": gamma_img,
        "ML": ml_img,
        "Log": log_img,
        "Bilateral": bilateral_img,
        "Sharpen": sharp_img,
        "Zero-DCE": zero_img,
        "Retinex": retinex_img,
        "CNN-Style": cnn_img
    }

    # Calculate metrics
    results = []

    for name, output in algorithms.items():

        psnr_value = psnr(img, output)
        ssim_value = calculate_ssim(img, output)
        mse_value = mse(img, output)
        brightness = brightness_score(output)
        contrast = contrast_score(output)
        sharpness = sharpness_score(output)

        score = (
            (0.20 * psnr_value)
            + (0.20 * (ssim_value * 100))
            + (0.20 * brightness)
            + (0.20 * contrast)
            + (0.20 * sharpness)
        )

        results.append({
            "Algorithm": name,
            "PSNR": round(psnr_value, 2),
            "SSIM": round(ssim_value, 4),
            "MSE": round(mse_value, 2),
            "Brightness": round(brightness, 2),
            "Contrast": round(contrast, 2),
            "Sharpness": round(sharpness, 2),
            "Final Score": round(score, 2)
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Final Score",
        ascending=False
    )

    # Best algorithm
    best_name = df.iloc[0]["Algorithm"]
    best_output = algorithms[best_name]

    return (
        best_output,
        best_name,
        df,
        hist_img,
        gamma_img,
        ml_img,
        log_img,
        bilateral_img,
        sharp_img,
        zero_img,
        retinex_img,
        cnn_img
    )


# =========================================================
# GRADIO INTERFACE
# =========================================================

interface = gr.Interface(
    fn=process_image,

    inputs=gr.Image(type="numpy", label="Upload Low-Light Image"),

    outputs=[
        gr.Image(label="Best Enhanced Output"),
        gr.Textbox(label="Best Algorithm"),
        gr.Dataframe(label="Comparison Table"),

        gr.Image(label="Histogram"),
        gr.Image(label="Gamma"),
        gr.Image(label="ML"),
        gr.Image(label="Log"),
        gr.Image(label="Bilateral"),
        gr.Image(label="Sharpen"),
        gr.Image(label="Zero-DCE"),
        gr.Image(label="Retinex"),
        gr.Image(label="CNN-Style")
    ],

    title="Low-Light Image Enhancement Using AI",

    description="Upload a low-light image and compare multiple enhancement algorithms automatically.",

   
)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == '__main__':
    interface.launch()