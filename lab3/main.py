"""
Face Detection, Feature Descriptors, Optical Flow & Background Subtraction
Minimal PyQt5 GUI with system-native look. Heavy operations in a background QThread.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox,
    QComboBox, QMessageBox, QSplitter, QSizePolicy, QTabWidget,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

import image_processing as proc


# ── Worker thread ───────────────────────────────────────────────────────────

class Worker(QThread):
    """Runs a callable in a background thread and emits the result."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── Helpers ─────────────────────────────────────────────────────────────────

def cv2pix(img, max_w, max_h):
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        qi = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi.copy()).scaled(
        max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class ImgLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img = None

    def set_image(self, img):
        self._img = img
        self._repaint()

    def _repaint(self):
        if self._img is None:
            return
        self.setPixmap(cv2pix(self._img, self.width(), self.height()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._repaint()


# ── Tab 1: Face Analysis ───────────────────────────────────────────────────

class FaceTab(QWidget):
    def __init__(self):
        super().__init__()
        self.src = None
        self.dst = None
        self._worker = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        # Left panel
        left = QWidget()
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        b_open = QPushButton("Open Image...")
        b_open.clicked.connect(self._open)
        lv.addWidget(b_open)

        g = QGroupBox("Method")
        gl = QVBoxLayout(g)
        self.cmb = QComboBox()
        self.cmb.addItems([
            "Face Contours",
            "SIFT",
            "SURF",
            "ORB",
            "HOG",
        ])
        gl.addWidget(self.cmb)
        lv.addWidget(g)

        self.b_apply = QPushButton("Apply")
        self.b_apply.clicked.connect(self._apply)
        lv.addWidget(self.b_apply)

        b_save = QPushButton("Save Result...")
        b_save.clicked.connect(self._save)
        lv.addWidget(b_save)

        lv.addStretch()
        root.addWidget(left)

        # Right panel
        sp = QSplitter(Qt.Horizontal)
        self.lbl_src = ImgLabel("Original")
        self.lbl_dst = ImgLabel("Result")
        sp.addWidget(self.lbl_src)
        sp.addWidget(self.lbl_dst)
        root.addWidget(sp, 1)

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not p:
            return
        img = proc.load_image(p, max_dim=1600)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot read file.")
            return
        self.src = img
        self.dst = None
        self.lbl_src.set_image(img)
        self.lbl_dst.set_image(None)
        self.lbl_dst.setText("Result")

    def _save(self):
        if self.dst is None:
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save", "result.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if p:
            cv2.imwrite(p, self.dst)

    def _set_busy(self, busy):
        self.b_apply.setEnabled(not busy)
        self.b_apply.setText("Processing..." if busy else "Apply")
        self.cmb.setEnabled(not busy)

    def _apply(self):
        if self.src is None:
            QMessageBox.information(self, "", "Open an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        img = self.src.copy()
        i = self.cmb.currentIndex()

        def task():
            if i == 0:
                return proc.face_contours(img)
            elif i == 1:
                return proc.sift_features(img)
            elif i == 2:
                return proc.surf_features(img)
            elif i == 3:
                return proc.orb_features(img)
            elif i == 4:
                return proc.hog_features(img)

        self._set_busy(True)
        self.lbl_dst.setText("Processing...")

        self._worker = Worker(task)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._set_busy(False)
        if result is not None:
            self.dst = result
            self.lbl_dst.set_image(result)
        self._worker = None

    def _on_error(self, msg):
        self._set_busy(False)
        self.lbl_dst.setText("Result")
        QMessageBox.warning(self, "Error", msg)
        self._worker = None


# ── Tab 2: Video Analysis ──────────────────────────────────────────────────

class VideoTab(QWidget):
    METHODS = [
        "Optical Flow (Lucas-Kanade)",
        "Optical Flow (Farneback)",
        "Background Sub (MOG2)",
        "Background Sub (KNN)",
    ]

    def __init__(self):
        super().__init__()
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._running = False

        # Optical flow state
        self._prev_gray = None
        self._lk_points = None
        self._lk_mask = None

        # Background subtractor
        self._bg_sub = None

        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        # Left panel
        left = QWidget()
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        b_video = QPushButton("Open Video...")
        b_video.clicked.connect(self._open_video)
        lv.addWidget(b_video)

        b_cam = QPushButton("Open Camera")
        b_cam.clicked.connect(self._open_camera)
        lv.addWidget(b_cam)

        g = QGroupBox("Method")
        gl = QVBoxLayout(g)
        self.cmb = QComboBox()
        self.cmb.addItems(self.METHODS)
        gl.addWidget(self.cmb)
        lv.addWidget(g)

        self.b_start = QPushButton("Start")
        self.b_start.clicked.connect(self._start)
        lv.addWidget(self.b_start)

        self.b_stop = QPushButton("Stop")
        self.b_stop.clicked.connect(self._stop)
        self.b_stop.setEnabled(False)
        lv.addWidget(self.b_stop)

        lv.addStretch()
        root.addWidget(left)

        # Right panel
        sp = QSplitter(Qt.Horizontal)
        self.lbl_vid = ImgLabel("Video")
        self.lbl_res = ImgLabel("Result")
        sp.addWidget(self.lbl_vid)
        sp.addWidget(self.lbl_res)
        root.addWidget(sp, 1)

    def _open_video(self):
        self._stop()
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if not p:
            return
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open video file.")
            return
        self._set_capture(cap)

    def _open_camera(self):
        self._stop()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open camera.")
            return
        self._set_capture(cap)

    def _set_capture(self, cap):
        if self._cap is not None:
            self._cap.release()
        self._cap = cap
        self._reset_state()
        # Show first frame
        ret, frame = self._cap.read()
        if ret:
            frame = proc.resize_to_limit(frame, 960)
            self.lbl_vid.set_image(frame)
        self.lbl_res.set_image(None)
        self.lbl_res.setText("Result")

    def _reset_state(self):
        self._prev_gray = None
        self._lk_points = None
        self._lk_mask = None
        self._bg_sub = None

    def _start(self):
        if self._cap is None or not self._cap.isOpened():
            QMessageBox.information(self, "", "Open a video or camera first.")
            return
        self._reset_state()

        method = self.cmb.currentIndex()
        # Initialize background subtractor
        if method == 2:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
        elif method == 3:
            self._bg_sub = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400, detectShadows=True
            )

        self._running = True
        self.b_start.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.cmb.setEnabled(False)
        self._timer.start(30)

    def _stop(self):
        self._timer.stop()
        self._running = False
        self.b_start.setEnabled(True)
        self.b_stop.setEnabled(False)
        self.cmb.setEnabled(True)

    def _tick(self):
        if self._cap is None or not self._cap.isOpened():
            self._stop()
            return
        ret, frame = self._cap.read()
        if not ret:
            self._stop()
            return

        frame = proc.resize_to_limit(frame, 960)
        self.lbl_vid.set_image(frame)

        method = self.cmb.currentIndex()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == 0:
            result = self._process_lk(frame, curr_gray)
        elif method == 1:
            result = self._process_farneback(curr_gray)
        elif method in (2, 3):
            result = self._process_bg_sub(frame)
        else:
            result = frame

        self._prev_gray = curr_gray
        self.lbl_res.set_image(result)

    def _process_lk(self, frame, curr_gray):
        """Lucas-Kanade sparse optical flow."""
        if self._prev_gray is None:
            self._prev_gray = curr_gray
            self._lk_points = proc.find_good_features(curr_gray)
            self._lk_mask = np.zeros_like(frame)
            return frame

        result_pair = proc.compute_optical_flow_lk(self._prev_gray, curr_gray, self._lk_points)
        vis = frame.copy()

        if result_pair is not None:
            old_pts, new_pts = result_pair
            for (new, old) in zip(new_pts, old_pts):
                a = tuple(new.astype(int).ravel())
                b = tuple(old.astype(int).ravel())
                self._lk_mask = cv2.line(self._lk_mask, b, a, (0, 255, 0), 2)
                vis = cv2.circle(vis, a, 4, (0, 0, 255), -1)
            vis = cv2.add(vis, self._lk_mask)
            self._lk_points = new_pts.reshape(-1, 1, 2)

            # Re-detect features periodically
            if len(self._lk_points) < 30:
                new_features = proc.find_good_features(curr_gray)
                if new_features is not None:
                    self._lk_points = np.vstack([self._lk_points, new_features])
                    self._lk_mask = np.zeros_like(frame)
        else:
            # Lost all points, re-detect
            self._lk_points = proc.find_good_features(curr_gray)
            self._lk_mask = np.zeros_like(frame)

        return vis

    def _process_farneback(self, curr_gray):
        """Farneback dense optical flow."""
        if self._prev_gray is None:
            self._prev_gray = curr_gray
            return np.zeros((*curr_gray.shape, 3), dtype=np.uint8)

        return proc.compute_optical_flow_farneback(self._prev_gray, curr_gray)

    def _process_bg_sub(self, frame):
        """Background subtraction with morphological cleanup."""
        if self._bg_sub is None:
            return frame

        fg_mask = self._bg_sub.apply(frame)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Threshold to remove shadows (shadows have value 127 in MOG2/KNN)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Draw bounding rects on moving objects
        result = frame.copy()
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show mask in top-left corner
        mask_small = cv2.resize(fg_mask, (160, 120))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        result[0:120, 0:160] = mask_bgr

        return result

    def closeEvent(self, event):
        self._stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


# ── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection, Optical Flow & Background Subtraction")
        self.resize(1024, 640)

        tabs = QTabWidget()
        self.face_tab = FaceTab()
        self.video_tab = VideoTab()
        tabs.addTab(self.face_tab, "Face Analysis")
        tabs.addTab(self.video_tab, "Video Analysis")
        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        self.video_tab.closeEvent(event)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
