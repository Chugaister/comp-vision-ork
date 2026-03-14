"""
image_processing.py — Image analysis and contrast enhancement utilities.

Uses OpenCV for histogram equalization (global and adaptive CLAHE).
"""

import cv2
import numpy as np
from typing import NamedTuple


# ──────────────────────────── Data Structures ──────────────────────────── #

class ImageCharacteristics(NamedTuple):
    width: int
    height: int
    channels: int
    dtype: str
    min_val: int
    max_val: int
    mean_val: float
    std_val: float
    median_val: float
    dynamic_range: int
    total_pixels: int


# ──────────────────────────── Analysis ──────────────────────────── #

def compute_characteristics(img: np.ndarray) -> ImageCharacteristics:
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return ImageCharacteristics(
        width=w, height=h, channels=channels, dtype=str(img.dtype),
        min_val=int(img.min()), max_val=int(img.max()),
        mean_val=round(float(img.mean()), 2),
        std_val=round(float(img.std()), 2),
        median_val=round(float(np.median(img)), 2),
        dynamic_range=int(img.max()) - int(img.min()),
        total_pixels=h * w,
    )


def compute_brightness(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale brightness values (1-D array)."""
    if img.ndim == 2:
        return img.ravel()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.ravel()


def compute_histogram(brightness: np.ndarray, bins: int = 256) -> np.ndarray:
    hist, _ = np.histogram(brightness, bins=bins, range=(0, 256))
    return hist


# ──────────────────────────── Contrast Enhancement ──────────────────────────── #

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization via cv2.equalizeHist.
    For colour images, equalizes the L channel in LAB space."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def adaptive_histogram_equalization(img: np.ndarray,
                                    clip_limit: float = 2.0,
                                    grid_size: int = 8) -> np.ndarray:
    """CLAHE via cv2.createCLAHE.
    For colour images, applies CLAHE to the L channel in LAB space."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))

    if img.ndim == 2:
        return clahe.apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def qimage_to_numpy(qimg) -> np.ndarray:
    from PyQt5.QtGui import QImage
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 4)
    arr = np.array(ptr, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3].copy()


def numpy_to_qimage(arr: np.ndarray):
    from PyQt5.QtGui import QImage
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = arr.shape
    if ch == 3:
        return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    raise ValueError(f"Unsupported channel count: {ch}")