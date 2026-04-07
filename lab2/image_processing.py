"""
processing.py — Reusable image processing functions (OpenCV + NumPy only).

Filtering:
    - gaussian_blur
    - median_blur
    - bilateral_filter
    - sharpen_unsharp_mask
    - sharpen_laplacian

Segmentation:
    - threshold_binary
    - threshold_otsu
    - watershed_segmentation
    - grabcut_segmentation

Utilities:
    - ensure_odd
    - load_image
    - resize_to_limit
"""

import cv2
import numpy as np


# ── Utilities ───────────────────────────────────────────────────────────────


def ensure_odd(v: int) -> int:
    """Ensure value is odd (required for many OpenCV kernel sizes)."""
    return v if v % 2 == 1 else v + 1


def load_image(path: str, max_dim: int = 0) -> np.ndarray | None:
    """Load BGR image from disk. Optionally downscale if largest side > max_dim.

    Returns None if file cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if max_dim > 0:
        img = resize_to_limit(img, max_dim)
    return img


def resize_to_limit(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so its largest dimension ≤ max_dim. No-op if already fits."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# ── Filtering ───────────────────────────────────────────────────────────────


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur.

    Args:
        img:    Input BGR image.
        ksize:  Kernel size (will be forced to odd).
        sigma:  Gaussian σ. 0 = auto from ksize.
    """
    k = ensure_odd(ksize)
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply median filter.

    Args:
        img:    Input BGR image.
        ksize:  Kernel size (will be forced to odd).
    """
    k = ensure_odd(ksize)
    return cv2.medianBlur(img, k)


def bilateral_filter(
    img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Apply bilateral filter (edge-preserving smoothing).

    Args:
        img:          Input BGR image.
        d:            Diameter of pixel neighbourhood.
        sigma_color:  Filter sigma in the color space.
        sigma_space:  Filter sigma in the coordinate space.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def sharpen_unsharp_mask(
    img: np.ndarray, amount: float = 1.5, radius: int = 5, sigma: float = 0
) -> np.ndarray:
    """Sharpen via Unsharp Mask: result = img + amount * (img − blurred).

    Args:
        img:     Input BGR image.
        amount:  Sharpening strength (1.0 = 100%).
        radius:  Gaussian kernel size for the blur pass (forced to odd).
        sigma:   Gaussian σ for the blur pass. 0 = auto.
    """
    r = ensure_odd(radius)
    blurred = cv2.GaussianBlur(img, (r, r), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def sharpen_laplacian(img: np.ndarray) -> np.ndarray:
    """Sharpen by adding the Laplacian edge map back onto the original.

    Works on BGR images; converts internally to grayscale for Laplacian
    then merges the edge energy back into all three channels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap), 0, 255))
    lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1, lap_bgr, 1, 0)


# ── Segmentation ────────────────────────────────────────────────────────────


def threshold_binary(img: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Simple binary threshold on a grayscale conversion of the input.

    Returns a single-channel uint8 image (0 or 255).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return result


def threshold_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu's automatic threshold. Pre-blurs with Gaussian (5×5).

    Returns:
        (binary_image, computed_threshold)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result, t


def watershed_segmentation(
    img: np.ndarray,
    fg_threshold_ratio: float = 0.5,
    morph_iterations: int = 2,
    dilate_iterations: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Automatic Watershed segmentation via distance-transform markers.

    Pipeline:
        1. Otsu threshold (inverted) → morphological opening
        2. Sure background via dilation
        3. Sure foreground via distance transform + threshold
        4. Markers from connected components
        5. cv2.watershed

    Args:
        img:                   Input BGR image.
        fg_threshold_ratio:    Fraction of max distance for sure-foreground.
        morph_iterations:      Iterations for morphological opening.
        dilate_iterations:     Iterations to dilate for sure-background.
        seed:                  Random seed for segment colours.

    Returns:
        (coloured_overlay, markers_map)
        - coloured_overlay: BGR image, each segment blended with a random colour,
          boundaries drawn in red.
        - markers_map: int32 array with segment labels (−1 = boundary).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iterations)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_threshold_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # Build coloured overlay
    unique = np.unique(markers)
    rng = np.random.default_rng(seed)
    colors = rng.integers(60, 255, size=(len(unique) + 2, 3), dtype=np.uint8)
    overlay = np.zeros_like(img)
    for i, u in enumerate(unique):
        if u <= 0:
            continue
        overlay[markers == u] = colors[i]

    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    result[markers == -1] = [0, 0, 255]
    return result, markers


def grabcut_segmentation(
    img: np.ndarray,
    iterations: int = 5,
    margin_x_ratio: float = 0.1,
    margin_y_ratio: float = 0.1,
    bg_darken: float = 0.25,
    draw_rect: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """GrabCut foreground extraction with an automatic central rectangle.

    Args:
        img:             Input BGR image.
        iterations:      Number of GrabCut iterations.
        margin_x_ratio:  Horizontal margin as fraction of width (each side).
        margin_y_ratio:  Vertical margin as fraction of height (each side).
        bg_darken:       Multiplier for background pixels (0 = black, 1 = unchanged).
        draw_rect:       Whether to draw the initial rectangle on the result.

    Returns:
        (result_image, foreground_mask)
        - result_image: BGR with background darkened.
        - foreground_mask: uint8 mask, 255 = foreground.
    """
    h, w = img.shape[:2]
    mx = int(w * margin_x_ratio)
    my = int(h * margin_y_ratio)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    result = img.copy()
    result[fg_mask == 0] = (result[fg_mask == 0] * bg_darken).astype(np.uint8)

    if draw_rect:
        cv2.rectangle(
            result,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            (0, 255, 0),
            2,
        )

    return result, fg_mask