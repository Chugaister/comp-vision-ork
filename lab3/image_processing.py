"""
image_processing.py — Face detection, feature descriptors, optical flow & background subtraction.

Face Detection:
    - detect_faces
    - face_contours

Feature Descriptors (per-face ROI):
    - sift_features
    - surf_features
    - orb_features
    - hog_features

Optical Flow:
    - find_good_features
    - compute_optical_flow_lk
    - compute_optical_flow_farneback

Utilities:
    - load_image
    - resize_to_limit
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


# ── Face Detection ──────────────────────────────────────────────────────────


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
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return []
    return faces.tolist()


def face_contours(img: np.ndarray) -> np.ndarray:
    """Detect faces, apply Canny edge detection in each face ROI, draw contours."""
    result = img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Offset contours to original image coordinates
        for c in contours:
            c[:, :, 0] += x
            c[:, :, 1] += y
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

    return result


# ── Feature Descriptors (per-face ROI) ──────────────────────────────────────


def sift_features(img: np.ndarray) -> np.ndarray:
    """Detect SIFT keypoints in each face ROI and draw rich keypoints (red)."""
    result = img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        kps, _ = sift.detectAndCompute(roi, None)
        # Draw keypoints on a sub-image, then paste back
        roi_bgr = result[y:y + h, x:x + w]
        cv2.drawKeypoints(roi_bgr, kps, roi_bgr,
                          color=(0, 0, 255),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result


def surf_features(img: np.ndarray) -> np.ndarray:
    """Detect SURF keypoints in each face ROI (requires opencv-contrib).

    Raises RuntimeError if SURF is not available.
    """
    if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "SURF_create"):
        raise RuntimeError(
            "SURF is not available in this OpenCV build.\n"
            "Install opencv-contrib-python to enable SURF support."
        )

    result = img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(400)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        kps, _ = surf.detectAndCompute(roi, None)
        roi_bgr = result[y:y + h, x:x + w]
        cv2.drawKeypoints(roi_bgr, kps, roi_bgr,
                          color=(0, 255, 255),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result


def orb_features(img: np.ndarray) -> np.ndarray:
    """Detect ORB keypoints in each face ROI and draw rich keypoints (magenta)."""
    result = img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(500)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        kps, _ = orb.detectAndCompute(roi, None)
        roi_bgr = result[y:y + h, x:x + w]
        cv2.drawKeypoints(roi_bgr, kps, roi_bgr,
                          color=(255, 0, 255),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return result


def hog_features(img: np.ndarray) -> np.ndarray:
    """Compute HOG-like visualization in each face ROI.

    Sobel gradients -> magnitude heatmap overlay + cell-level orientation lines.
    """
    result = img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w].astype(np.float32)

        # Sobel gradients
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)

        # Magnitude heatmap overlay
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
        roi_bgr = result[y:y + h, x:x + w]
        cv2.addWeighted(roi_bgr, 0.5, heatmap, 0.5, 0, dst=roi_bgr)

        # Cell-level orientation lines
        cell_size = 8
        rh, rw = roi.shape[:2]
        for cy in range(0, rh - cell_size + 1, cell_size):
            for cx in range(0, rw - cell_size + 1, cell_size):
                cell_mag = mag[cy:cy + cell_size, cx:cx + cell_size]
                cell_angle = angle[cy:cy + cell_size, cx:cx + cell_size]
                avg_mag = cell_mag.mean()
                if avg_mag < 10:
                    continue
                avg_angle = np.arctan2(
                    np.mean(np.sin(cell_angle)),
                    np.mean(np.cos(cell_angle))
                )
                center_x = x + cx + cell_size // 2
                center_y = y + cy + cell_size // 2
                length = min(cell_size // 2, max(2, int(avg_mag / mag.max() * cell_size // 2)))
                dx = int(length * np.cos(avg_angle))
                dy = int(length * np.sin(avg_angle))
                cv2.line(result,
                         (center_x - dx, center_y - dy),
                         (center_x + dx, center_y + dy),
                         (255, 255, 255), 1)

    return result


# ── Optical Flow ────────────────────────────────────────────────────────────


def find_good_features(gray: np.ndarray) -> np.ndarray | None:
    """Find good features to track using Shi-Tomasi corner detection.

    Returns Nx1x2 float32 array of points, or None.
    """
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
    )


def compute_optical_flow_lk(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute sparse optical flow with Lucas-Kanade.

    Returns (old_points, new_points) for successfully tracked points, or None.
    """
    if prev_pts is None or len(prev_pts) == 0:
        return None

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    good_mask = status.ravel() == 1
    if not np.any(good_mask):
        return None
    return prev_pts[good_mask], next_pts[good_mask]


def compute_optical_flow_farneback(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """Compute dense optical flow with Farneback method.

    Returns BGR visualization (HSV-based: hue=direction, value=magnitude).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
