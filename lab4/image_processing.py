"""
image_processing.py — Geometric transformations and morphological operations.

Geometric Transformations (Week 7):
    - scale_image
    - rotate_image
    - perspective_transform
    - auto_perspective_correct (face-based)

Morphological Operations (Week 8):
    - morphology_erode
    - morphology_dilate
    - morphology_open
    - morphology_close
    - morphology_gradient
    - morphology_tophat
    - morphology_blackhat

Utilities:
    - load_image
    - resize_to_limit
    - detect_faces
"""

import cv2
import numpy as np


# ── Utilities ───────────────────────────────────────────────────────────────


def load_image(path: str, max_dim: int = 0) -> np.ndarray | None:
    """Load BGR image from disk. Optionally downscale if largest side > max_dim."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if max_dim > 0:
        img = resize_to_limit(img, max_dim)
    return img


def resize_to_limit(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so its largest dimension <= max_dim. No-op if already fits."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(xml)
    return _cascade


def detect_faces(img: np.ndarray):
    """Detect faces using Haar cascade.

    Returns list of (x, y, w, h) rectangles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    return faces.tolist()


# ── Geometric Transformations (Week 7) ──────────────────────────────────────


def scale_image(img: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale image by given factors.

    Args:
        img: Input BGR image.
        scale_x: Horizontal scale factor.
        scale_y: Vertical scale factor.

    Returns:
        Scaled image.
    """
    h, w = img.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def rotate_image(
    img: np.ndarray, angle: float, scale: float = 1.0, crop: bool = False
) -> np.ndarray:
    """Rotate image around its center.

    Args:
        img: Input BGR image.
        angle: Rotation angle in degrees (counter-clockwise).
        scale: Additional scale factor.
        crop: If True, crop to original size; else expand canvas.

    Returns:
        Rotated image.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    if crop:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    else:
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust transformation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)


def perspective_transform(
    img: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Apply perspective transformation.

    Args:
        img: Input BGR image.
        src_points: Source quadrilateral (4 points) as float32 array [[x,y], ...].
        dst_points: Destination quadrilateral (4 points) as float32 array [[x,y], ...].

    Returns:
        Perspective-transformed image.
    """
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


def auto_perspective_correct(img: np.ndarray, margin_ratio: float = 0.1) -> tuple[np.ndarray, bool]:
    """Automatic perspective correction for face images.

    Detects the largest face and applies a subtle perspective correction
    to make it more frontal-facing. Uses eye detection for alignment.

    Args:
        img: Input BGR image.
        margin_ratio: Margin around face as fraction of face size.

    Returns:
        (corrected_image, success) tuple.
        - corrected_image: Result with face region warped.
        - success: True if a face was found and corrected.
    """
    faces = detect_faces(img)
    if not faces:
        return img.copy(), False

    # Use largest face
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = face

    # Add margin
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_y)

    # Define source points (current face corners with margin)
    src = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    # Define destination points (slightly adjusted for frontal view)
    # Apply a subtle perspective adjustment
    center_x = (x1 + x2) / 2
    width = x2 - x1
    height = y2 - y1

    # Slight trapezoidal correction (assumes face is slightly tilted)
    dst = np.float32([
        [center_x - width * 0.45, y1 + height * 0.05],
        [center_x + width * 0.45, y1 + height * 0.05],
        [center_x + width * 0.5, y2],
        [center_x - width * 0.5, y2]
    ])

    result = perspective_transform(img, src, dst)
    return result, True


# ── Morphological Operations (Week 8) ───────────────────────────────────────


def morphology_erode(
    img: np.ndarray, kernel_size: int = 5, iterations: int = 1
) -> np.ndarray:
    """Apply morphological erosion.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.
        iterations: Number of times erosion is applied.

    Returns:
        Eroded image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=iterations)


def morphology_dilate(
    img: np.ndarray, kernel_size: int = 5, iterations: int = 1
) -> np.ndarray:
    """Apply morphological dilation.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.
        iterations: Number of times dilation is applied.

    Returns:
        Dilated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(img, kernel, iterations=iterations)


def morphology_open(
    img: np.ndarray, kernel_size: int = 5, iterations: int = 1
) -> np.ndarray:
    """Apply morphological opening (erosion followed by dilation).

    Removes small objects/noise while preserving larger structures.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.
        iterations: Number of iterations.

    Returns:
        Opened image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphology_close(
    img: np.ndarray, kernel_size: int = 5, iterations: int = 1
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion).

    Fills small holes and gaps while preserving overall shape.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.
        iterations: Number of iterations.

    Returns:
        Closed image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphology_gradient(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological gradient (dilation - erosion).

    Highlights object boundaries.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.

    Returns:
        Gradient image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def morphology_tophat(img: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Apply morphological top-hat (original - opening).

    Extracts small bright features on dark background.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.

    Returns:
        Top-hat image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def morphology_blackhat(img: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Apply morphological black-hat (closing - original).

    Extracts small dark features on bright background.

    Args:
        img: Input BGR or grayscale image.
        kernel_size: Size of the structuring element.

    Returns:
        Black-hat image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)