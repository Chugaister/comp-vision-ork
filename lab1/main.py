import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox,
    QSlider, QSpinBox, QSplitter, QSizePolicy, QFrame,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import image_processing as ip


class ImageLabel(QLabel):

    def __init__(self, placeholder: str = "No image"):
        super().__init__()
        self._pixmap = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.Box)
        self.setText(self._placeholder)

    def set_pixmap(self, pm: QPixmap | None):
        self._pixmap = pm
        if pm is None:
            self.setText(self._placeholder)
        else:
            self._rescale()

    def _rescale(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Contrast Analyzer")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 850)

        self._original_np: np.ndarray | None = None
        self._processed_np: np.ndarray | None = None

        self._build_ui()
        self.statusBar().showMessage("Ready — open an image to begin")

    # ── UI Construction ───────────────────────────────────────────── #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- Top toolbar row ----
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("Open Image…")
        self.btn_open.setMinimumWidth(140)
        self.btn_open.clicked.connect(self._open_image)

        self.btn_save = QPushButton("Save Result…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_image)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._reset)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_save)
        toolbar.addWidget(self.btn_reset)
        toolbar.addStretch()

        self.lbl_filename = QLabel("")
        toolbar.addWidget(self.lbl_filename)

        root.addLayout(toolbar)

        # ---- Main splitter: left panel | right panel ----
        splitter = QSplitter(Qt.Horizontal)

        # -- Left: images --
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        images_row = QHBoxLayout()
        # Original
        orig_group = QGroupBox("Original")
        orig_lay = QVBoxLayout(orig_group)
        self.img_original = ImageLabel("Open an image\n(PNG, JPG, BMP)")
        orig_lay.addWidget(self.img_original)
        images_row.addWidget(orig_group)

        # Processed
        proc_group = QGroupBox("Processed")
        proc_lay = QVBoxLayout(proc_group)
        self.img_processed = ImageLabel("Apply an enhancement\nmethod")
        proc_lay.addWidget(self.img_processed)
        images_row.addWidget(proc_group)

        left_lay.addLayout(images_row, stretch=3)

        # Histogram
        hist_group = QGroupBox("Brightness Histogram")
        hist_lay = QVBoxLayout(hist_group)
        self.hist_label = ImageLabel("Histogram will appear here")
        self.hist_label.setMinimumHeight(180)
        hist_lay.addWidget(self.hist_label)
        left_lay.addWidget(hist_group, stretch=2)

        splitter.addWidget(left)

        # -- Right: controls & characteristics --
        right = QWidget()
        right.setMaximumWidth(340)
        right.setMinimumWidth(280)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        # Characteristics
        char_group = QGroupBox("Image Characteristics")
        char_lay = QVBoxLayout(char_group)
        self.lbl_chars = QLabel("—")
        self.lbl_chars.setWordWrap(True)
        self.lbl_chars.setTextFormat(Qt.RichText)
        self.lbl_chars.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        char_lay.addWidget(self.lbl_chars)
        right_lay.addWidget(char_group)

        # Enhancement controls
        enh_group = QGroupBox("Contrast Enhancement")
        enh_lay = QVBoxLayout(enh_group)

        self.btn_hist_eq = QPushButton("Histogram Equalization")
        self.btn_hist_eq.setEnabled(False)
        self.btn_hist_eq.clicked.connect(self._apply_hist_eq)
        enh_lay.addWidget(self.btn_hist_eq)

        # CLAHE controls
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        enh_lay.addWidget(sep)

        clahe_title = QLabel("Adaptive (CLAHE)")
        clahe_title.setAlignment(Qt.AlignLeft)
        font = clahe_title.font()
        font.setBold(True)
        clahe_title.setFont(font)
        enh_lay.addWidget(clahe_title)

        # Clip limit
        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("Clip limit:"))
        self.slider_clip = QSlider(Qt.Horizontal)
        self.slider_clip.setRange(10, 100)
        self.slider_clip.setValue(20)
        self.slider_clip.setTickInterval(10)
        self.lbl_clip = QLabel("2.0")
        self.lbl_clip.setMinimumWidth(30)
        self.slider_clip.valueChanged.connect(
            lambda v: self.lbl_clip.setText(f"{v / 10:.1f}")
        )
        clip_row.addWidget(self.slider_clip)
        clip_row.addWidget(self.lbl_clip)
        enh_lay.addLayout(clip_row)

        # Grid size
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid size:"))
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(2, 16)
        self.spin_grid.setValue(8)
        grid_row.addWidget(self.spin_grid)
        enh_lay.addLayout(grid_row)

        self.btn_clahe = QPushButton("Apply CLAHE")
        self.btn_clahe.setEnabled(False)
        self.btn_clahe.clicked.connect(self._apply_clahe)
        enh_lay.addWidget(self.btn_clahe)

        right_lay.addWidget(enh_group)

        # Processed characteristics
        pchar_group = QGroupBox("Processed Characteristics")
        pchar_lay = QVBoxLayout(pchar_group)
        self.lbl_pchars = QLabel("—")
        self.lbl_pchars.setWordWrap(True)
        self.lbl_pchars.setTextFormat(Qt.RichText)
        self.lbl_pchars.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        pchar_lay.addWidget(self.lbl_pchars)
        right_lay.addWidget(pchar_group)

        right_lay.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if not path:
            return

        qimg = QImage(path)
        if qimg.isNull():
            self.statusBar().showMessage(f"Failed to load: {path}")
            return

        self._original_np = ip.qimage_to_numpy(qimg)
        self._processed_np = None

        pm = QPixmap.fromImage(ip.numpy_to_qimage(self._original_np))
        self.img_original.set_pixmap(pm)
        self.img_processed.set_pixmap(None)

        self._update_characteristics(self._original_np, self.lbl_chars)
        self.lbl_pchars.setText("—")
        self._update_histogram()

        self.btn_hist_eq.setEnabled(True)
        self.btn_clahe.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)

        short = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
        self.lbl_filename.setText(short)
        self.statusBar().showMessage(f"Loaded: {short}")

    def _save_image(self):
        if self._processed_np is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "enhanced.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            qimg = ip.numpy_to_qimage(self._processed_np)
            qimg.save(path)
            self.statusBar().showMessage(f"Saved: {path}")

    def _reset(self):
        if self._original_np is None:
            return
        self._processed_np = None
        self.img_processed.set_pixmap(None)
        self.lbl_pchars.setText("—")
        self.btn_save.setEnabled(False)
        self._update_histogram()
        self.statusBar().showMessage("Reset to original")

    def _apply_hist_eq(self):
        if self._original_np is None:
            return
        self.statusBar().showMessage("Applying Histogram Equalization…")
        QApplication.processEvents()
        self._processed_np = ip.histogram_equalization(self._original_np)
        self._show_processed()
        self.statusBar().showMessage("Histogram Equalization applied")

    def _apply_clahe(self):
        if self._original_np is None:
            return
        clip = self.slider_clip.value() / 10.0
        grid = self.spin_grid.value()
        self.statusBar().showMessage(
            f"Applying CLAHE (clip={clip}, grid={grid}×{grid})…"
        )
        QApplication.processEvents()
        self._processed_np = ip.adaptive_histogram_equalization(
            self._original_np, clip_limit=clip, grid_size=grid
        )
        self._show_processed()
        self.statusBar().showMessage(
            f"CLAHE applied (clip={clip}, grid={grid}×{grid})"
        )

    # ── Internal helpers ──────────────────────────────────────────── #

    def _show_processed(self):
        pm = QPixmap.fromImage(ip.numpy_to_qimage(self._processed_np))
        self.img_processed.set_pixmap(pm)
        self._update_characteristics(self._processed_np, self.lbl_pchars)
        self._update_histogram()
        self.btn_save.setEnabled(True)

    @staticmethod
    def _format_chars(c: ip.ImageCharacteristics) -> str:
        ch_name = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(c.channels, f"{c.channels}ch")
        return (
            f"<table cellpadding='2'>"
            f"<tr><td>Size:</td><td><b>{c.width} × {c.height}</b></td></tr>"
            f"<tr><td>Pixels:</td><td>{c.total_pixels:,}</td></tr>"
            f"<tr><td>Channels:</td><td>{ch_name}</td></tr>"
            f"<tr><td>Dtype:</td><td>{c.dtype}</td></tr>"
            f"<tr><td>Min / Max:</td><td>{c.min_val} / {c.max_val}</td></tr>"
            f"<tr><td>Dynamic range:</td><td>{c.dynamic_range}</td></tr>"
            f"<tr><td>Mean:</td><td>{c.mean_val}</td></tr>"
            f"<tr><td>Median:</td><td>{c.median_val}</td></tr>"
            f"<tr><td>Std dev:</td><td>{c.std_val}</td></tr>"
            f"</table>"
        )

    def _update_characteristics(self, img: np.ndarray, label: QLabel):
        c = ip.compute_characteristics(img)
        label.setText(self._format_chars(c))

    def _update_histogram(self):
        if self._original_np is None:
            return

        fig = Figure(figsize=(7, 2.6), dpi=120)
        ax = fig.add_subplot(111)

        # Original histogram
        bright_orig = ip.compute_brightness(self._original_np)
        hist_orig = ip.compute_histogram(bright_orig)
        ax.fill_between(range(256), hist_orig, alpha=0.45, color="steelblue", label="Original")
        ax.plot(range(256), hist_orig, color="steelblue", linewidth=0.8)

        # Processed histogram (if available)
        if self._processed_np is not None:
            bright_proc = ip.compute_brightness(self._processed_np)
            hist_proc = ip.compute_histogram(bright_proc)
            ax.fill_between(range(256), hist_proc, alpha=0.40, color="forestgreen", label="Processed")
            ax.plot(range(256), hist_proc, color="forestgreen", linewidth=0.8)

        ax.set_xlim(0, 255)
        ax.set_xlabel("Brightness", fontsize=9)
        ax.set_ylabel("Pixel count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout(pad=1.0)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        qimg = QImage(buf, w, h, QImage.Format_RGBA8888).copy()
        self.hist_label.set_pixmap(QPixmap.fromImage(qimg))
        plt.close(fig)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
