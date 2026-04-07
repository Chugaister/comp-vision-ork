"""
Image Processing — Filtering & Segmentation
Minimal PyQt5 GUI with system-native look. Processing in processing.py.
Heavy operations run in a background QThread to keep the UI responsive.
"""

import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QSlider,
    QComboBox, QSpinBox, QMessageBox, QSplitter, QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np

import image_processing as proc


# ── Worker thread ───────────────────────────────────────────────────────────

class Worker(QThread):
    """Runs a callable in a background thread and emits the result."""
    finished = pyqtSignal(object)  # emits the result image (ndarray or None)
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


# ── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.resize(960, 600)
        self.src = None
        self.dst = None
        self._worker = None
        self._build()

    def _build(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QHBoxLayout(c)

        # ── Left panel ──
        left = QWidget()
        left.setFixedWidth(260)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        # File buttons
        b_open = QPushButton("Open…")
        b_open.clicked.connect(self._open)
        b_save = QPushButton("Save…")
        b_save.clicked.connect(self._save)
        row = QHBoxLayout()
        row.addWidget(b_open)
        row.addWidget(b_save)
        lv.addLayout(row)

        # Method
        g1 = QGroupBox("Method")
        g1l = QVBoxLayout(g1)
        self.cmb = QComboBox()
        self.cmb.addItems([
            "Gaussian Blur", "Median Filter", "Bilateral Filter",
            "Sharpen — Unsharp Mask", "Sharpen — Laplacian",
            "Binary Threshold", "Otsu Threshold",
            "Watershed", "GrabCut",
        ])
        self.cmb.currentIndexChanged.connect(self._method_changed)
        g1l.addWidget(self.cmb)
        lv.addWidget(g1)

        # Parameters
        g2 = QGroupBox("Parameters")
        self._g2l = QVBoxLayout(g2)

        def _sp(label, lo, hi, val, step=1):
            w = QWidget()
            r = QHBoxLayout(w)
            r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(QLabel(label))
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            r.addWidget(s)
            self._g2l.addWidget(w)
            return w, s

        def _sl(label, lo, hi, val):
            w = QWidget()
            r = QHBoxLayout(w)
            r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(QLabel(label))
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            v = QLabel(str(val))
            v.setFixedWidth(28)
            s.valueChanged.connect(lambda x: v.setText(str(x)))
            r.addWidget(s)
            r.addWidget(v)
            self._g2l.addWidget(w)
            return w, s

        self.w_ksize, self.sp_ksize   = _sp("Kernel", 1, 51, 5, 2)
        self.w_sigma, self.sp_sigma   = _sp("Sigma", 0, 100, 1)
        self.w_d, self.sp_d           = _sp("Diameter", 1, 25, 9)
        self.w_sc, self.sp_sc         = _sp("σ Color", 1, 300, 75)
        self.w_ss, self.sp_ss         = _sp("σ Space", 1, 300, 75)
        self.w_amt, self.sp_amt       = _sp("Amount %", 1, 500, 150)
        self.w_rad, self.sp_rad       = _sp("Radius", 1, 51, 5, 2)
        self.w_thresh, self.sl_thresh = _sl("Threshold", 0, 255, 127)
        self.w_gciter, self.sp_gciter = _sp("Iterations", 1, 20, 5)

        self._param_map = {
            0: [self.w_ksize, self.w_sigma],
            1: [self.w_ksize],
            2: [self.w_d, self.w_sc, self.w_ss],
            3: [self.w_amt, self.w_rad, self.w_sigma],
            4: [],
            5: [self.w_thresh],
            6: [],
            7: [],
            8: [self.w_gciter],
        }
        self._all_params = [
            self.w_ksize, self.w_sigma, self.w_d, self.w_sc, self.w_ss,
            self.w_amt, self.w_rad, self.w_thresh, self.w_gciter,
        ]
        lv.addWidget(g2)

        self.b_apply = QPushButton("Apply")
        self.b_apply.clicked.connect(self._apply)
        lv.addWidget(self.b_apply)
        lv.addStretch()

        root.addWidget(left)

        # ── Right panel — images ──
        sp = QSplitter(Qt.Horizontal)
        self.lbl_src = ImgLabel("Original")
        self.lbl_dst = ImgLabel("Result")
        sp.addWidget(self.lbl_src)
        sp.addWidget(self.lbl_dst)
        root.addWidget(sp, 1)

        self._method_changed(0)

    # ── Visibility ──

    def _method_changed(self, idx):
        show = set(self._param_map.get(idx, []))
        for w in self._all_params:
            w.setVisible(w in show)

    # ── I/O ──

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open", "",
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

    # ── UI lock/unlock during processing ──

    def _set_busy(self, busy):
        self.b_apply.setEnabled(not busy)
        self.b_apply.setText("Processing…" if busy else "Apply")
        self.cmb.setEnabled(not busy)

    # ── Apply (runs processing in a worker thread) ──

    def _apply(self):
        if self.src is None:
            QMessageBox.information(self, "", "Open an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        img = self.src.copy()
        i = self.cmb.currentIndex()

        # Capture current parameter values (must read from GUI in main thread)
        ksize = self.sp_ksize.value()
        sigma = self.sp_sigma.value()
        d = self.sp_d.value()
        sc = self.sp_sc.value()
        ss = self.sp_ss.value()
        amt = self.sp_amt.value() / 100.0
        rad = self.sp_rad.value()
        thresh = self.sl_thresh.value()
        gciter = self.sp_gciter.value()

        # Build the processing callable
        def task():
            if i == 0:
                return proc.gaussian_blur(img, ksize, sigma)
            elif i == 1:
                return proc.median_blur(img, ksize)
            elif i == 2:
                return proc.bilateral_filter(img, d, sc, ss)
            elif i == 3:
                return proc.sharpen_unsharp_mask(img, amt, rad, sigma)
            elif i == 4:
                return proc.sharpen_laplacian(img)
            elif i == 5:
                return proc.threshold_binary(img, thresh)
            elif i == 6:
                r, _ = proc.threshold_otsu(img)
                return r
            elif i == 7:
                r, _ = proc.watershed_segmentation(img)
                return r
            elif i == 8:
                r, _ = proc.grabcut_segmentation(img, gciter)
                return r

        self._set_busy(True)
        self.lbl_dst.setText("Processing…")

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


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()