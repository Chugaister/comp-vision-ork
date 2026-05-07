import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox,
    QSlider, QSpinBox, QSplitter,
    QComboBox, QCheckBox, QTabWidget,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
import image_processing as ip


class ImageLabel(QLabel):

    def __init__(self, placeholder: str = "No image"):
        super().__init__()
        self._pixmap = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
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
        self.setWindowTitle("Lab 4 - Geometric Transformations & Morphological Operations")

        self._original_bgr: np.ndarray | None = None
        self._processed_bgr: np.ndarray | None = None

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
        self.img_processed = ImageLabel("Apply a transformation\nor morphological operation")
        proc_lay.addWidget(self.img_processed)
        images_row.addWidget(proc_group)

        left_lay.addLayout(images_row)
        splitter.addWidget(left)

        # -- Right: controls --
        right = QWidget()
        right_lay = QVBoxLayout(right)

        # Tab widget for operations
        tabs = QTabWidget()

        # === Tab 1: Geometric Transformations ===
        geo_tab = QWidget()
        geo_lay = QVBoxLayout(geo_tab)

        # Scale
        scale_group = QGroupBox("Scale")
        scale_lay = QVBoxLayout(scale_group)

        scale_x_row = QHBoxLayout()
        scale_x_row.addWidget(QLabel("Scale X:"))
        self.slider_scale_x = QSlider(Qt.Horizontal)
        self.slider_scale_x.setRange(10, 300)
        self.slider_scale_x.setValue(100)
        self.lbl_scale_x = QLabel("1.00")
        self.slider_scale_x.valueChanged.connect(
            lambda v: self._update_scale_x_label(v)
        )
        scale_x_row.addWidget(self.slider_scale_x)
        scale_x_row.addWidget(self.lbl_scale_x)
        scale_lay.addLayout(scale_x_row)

        scale_y_row = QHBoxLayout()
        scale_y_row.addWidget(QLabel("Scale Y:"))
        self.slider_scale_y = QSlider(Qt.Horizontal)
        self.slider_scale_y.setRange(10, 300)
        self.slider_scale_y.setValue(100)
        self.lbl_scale_y = QLabel("1.00")
        self.slider_scale_y.valueChanged.connect(
            lambda v: self._update_scale_y_label(v)
        )
        scale_y_row.addWidget(self.slider_scale_y)
        scale_y_row.addWidget(self.lbl_scale_y)
        scale_lay.addLayout(scale_y_row)

        self.btn_scale = QPushButton("Apply Scale")
        self.btn_scale.setEnabled(False)
        self.btn_scale.clicked.connect(self._apply_scale)
        scale_lay.addWidget(self.btn_scale)

        geo_lay.addWidget(scale_group)

        # Rotation
        rot_group = QGroupBox("Rotation")
        rot_lay = QVBoxLayout(rot_group)

        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Angle:"))
        self.slider_angle = QSlider(Qt.Horizontal)
        self.slider_angle.setRange(-180, 180)
        self.slider_angle.setValue(0)
        self.lbl_angle = QLabel("0°")
        self.slider_angle.valueChanged.connect(
            lambda v: self._update_angle_label(v)
        )
        angle_row.addWidget(self.slider_angle)
        angle_row.addWidget(self.lbl_angle)
        rot_lay.addLayout(angle_row)

        rot_scale_row = QHBoxLayout()
        rot_scale_row.addWidget(QLabel("Scale:"))
        self.slider_rot_scale = QSlider(Qt.Horizontal)
        self.slider_rot_scale.setRange(10, 200)
        self.slider_rot_scale.setValue(100)
        self.lbl_rot_scale = QLabel("1.00")
        self.slider_rot_scale.valueChanged.connect(
            lambda v: self._update_rot_scale_label(v)
        )
        rot_scale_row.addWidget(self.slider_rot_scale)
        rot_scale_row.addWidget(self.lbl_rot_scale)
        rot_lay.addLayout(rot_scale_row)

        self.check_crop = QCheckBox("Crop to original size")
        self.check_crop.setChecked(False)
        rot_lay.addWidget(self.check_crop)

        self.btn_rotate = QPushButton("Apply Rotation")
        self.btn_rotate.setEnabled(False)
        self.btn_rotate.clicked.connect(self._apply_rotation)
        rot_lay.addWidget(self.btn_rotate)

        geo_lay.addWidget(rot_group)

        # Perspective correction
        persp_group = QGroupBox("Perspective Correction")
        persp_lay = QVBoxLayout(persp_group)

        self.btn_auto_persp = QPushButton("Auto-correct Face Perspective")
        self.btn_auto_persp.setEnabled(False)
        self.btn_auto_persp.clicked.connect(self._apply_auto_perspective)
        persp_lay.addWidget(self.btn_auto_persp)

        geo_lay.addWidget(persp_group)
        geo_lay.addStretch()

        tabs.addTab(geo_tab, "Geometric")

        # Debug: verify sliders are independent objects
        print("=== Slider Debug Info ===")
        print(f"scale_x slider: {id(self.slider_scale_x)}")
        print(f"scale_y slider: {id(self.slider_scale_y)}")
        print(f"angle slider: {id(self.slider_angle)}")
        print(f"rot_scale slider: {id(self.slider_rot_scale)}")
        print(f"All unique: {len({id(self.slider_scale_x), id(self.slider_scale_y), id(self.slider_angle), id(self.slider_rot_scale)}) == 4}")
        print("========================")

        # === Tab 2: Morphological Operations ===
        morph_tab = QWidget()
        morph_lay = QVBoxLayout(morph_tab)

        operation_group = QGroupBox("Morphological Operation")
        operation_lay = QVBoxLayout(operation_group)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self.combo_morph_op = QComboBox()
        self.combo_morph_op.addItems([
            "Erosion",
            "Dilation",
            "Opening",
            "Closing",
            "Gradient",
            "Top Hat",
            "Black Hat",
        ])
        op_row.addWidget(self.combo_morph_op)
        operation_lay.addLayout(op_row)

        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel size:"))
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 31)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(5)
        kernel_row.addWidget(self.spin_kernel)
        operation_lay.addLayout(kernel_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterations:"))
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(1, 10)
        self.spin_iterations.setValue(1)
        iter_row.addWidget(self.spin_iterations)
        operation_lay.addLayout(iter_row)

        self.btn_morph = QPushButton("Apply Morphological Op")
        self.btn_morph.setEnabled(False)
        self.btn_morph.clicked.connect(self._apply_morphology)
        operation_lay.addWidget(self.btn_morph)

        morph_lay.addWidget(operation_group)

        # Info box
        info_label = QLabel(
            "Morphological Operations Info:\n"
            "Erosion: Shrinks bright regions\n"
            "Dilation: Expands bright regions\n"
            "Opening: Removes noise (erode→dilate)\n"
            "Closing: Fills gaps (dilate→erode)\n"
            "Gradient: Highlights edges\n"
            "Top Hat: Bright features extraction\n"
            "Black Hat: Dark features extraction"
        )
        morph_lay.addWidget(info_label)

        morph_lay.addStretch()

        tabs.addTab(morph_tab, "Morphological")

        right_lay.addWidget(tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ── Slider update handlers (with debug) ──────────────────────── #

    def _update_scale_x_label(self, v):
        print(f"[DEBUG] scale_x slider changed: {v}")
        self.lbl_scale_x.setText(f"{v / 100:.2f}")

    def _update_scale_y_label(self, v):
        print(f"[DEBUG] scale_y slider changed: {v}")
        self.lbl_scale_y.setText(f"{v / 100:.2f}")

    def _update_angle_label(self, v):
        print(f"[DEBUG] angle slider changed: {v}")
        self.lbl_angle.setText(f"{v}°")

    def _update_rot_scale_label(self, v):
        print(f"[DEBUG] rot_scale slider changed: {v}")
        self.lbl_rot_scale.setText(f"{v / 100:.2f}")

    # ── Actions ───────────────────────────────────────────────────── #

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if not path:
            return

        img = ip.load_image(path)
        if img is None:
            self.statusBar().showMessage(f"Failed to load: {path}")
            return

        self._original_bgr = img
        self._processed_bgr = None

        # Convert BGR to RGB for display
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pm = self._numpy_to_pixmap(rgb)
        self.img_original.set_pixmap(pm)
        self.img_processed.set_pixmap(None)

        # Enable controls
        self.btn_scale.setEnabled(True)
        self.btn_rotate.setEnabled(True)
        self.btn_auto_persp.setEnabled(True)
        self.btn_morph.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)

        short = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
        self.lbl_filename.setText(short)
        self.statusBar().showMessage(f"Loaded: {short}")

    def _save_image(self):
        if self._processed_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "result.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, self._processed_bgr)
            self.statusBar().showMessage(f"Saved: {path}")

    def _reset(self):
        if self._original_bgr is None:
            return
        self._processed_bgr = None
        self.img_processed.set_pixmap(None)
        self.btn_save.setEnabled(False)
        self.statusBar().showMessage("Reset to original")

    # ── Geometric Transformations ─────────────────────────────────── #

    def _apply_scale(self):
        if self._original_bgr is None:
            return
        scale_x = self.slider_scale_x.value() / 100.0
        scale_y = self.slider_scale_y.value() / 100.0

        self.statusBar().showMessage(f"Applying scale ({scale_x:.2f}, {scale_y:.2f})…")
        QApplication.processEvents()

        self._processed_bgr = ip.scale_image(self._original_bgr, scale_x, scale_y)
        self._show_processed()
        self.statusBar().showMessage(f"Scale applied: {scale_x:.2f} × {scale_y:.2f}")

    def _apply_rotation(self):
        if self._original_bgr is None:
            return
        angle = self.slider_angle.value()
        scale = self.slider_rot_scale.value() / 100.0
        crop = self.check_crop.isChecked()

        self.statusBar().showMessage(f"Applying rotation {angle}°…")
        QApplication.processEvents()

        self._processed_bgr = ip.rotate_image(self._original_bgr, angle, scale, crop)
        self._show_processed()
        crop_str = " (cropped)" if crop else ""
        self.statusBar().showMessage(f"Rotation applied: {angle}° × {scale:.2f}{crop_str}")

    def _apply_auto_perspective(self):
        if self._original_bgr is None:
            return

        self.statusBar().showMessage("Applying auto perspective correction…")
        QApplication.processEvents()

        result, success = ip.auto_perspective_correct(self._original_bgr)
        if not success:
            self.statusBar().showMessage("No face detected for perspective correction")
            return

        self._processed_bgr = result
        self._show_processed()
        self.statusBar().showMessage("Auto perspective correction applied")

    # ── Morphological Operations ──────────────────────────────────── #

    def _apply_morphology(self):
        if self._original_bgr is None:
            return

        op_name = self.combo_morph_op.currentText()
        kernel_size = self.spin_kernel.value()
        iterations = self.spin_iterations.value()

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.spin_kernel.setValue(kernel_size)

        self.statusBar().showMessage(f"Applying {op_name}…")
        QApplication.processEvents()

        img = self._original_bgr

        if op_name == "Erosion":
            result = ip.morphology_erode(img, kernel_size, iterations)
        elif op_name == "Dilation":
            result = ip.morphology_dilate(img, kernel_size, iterations)
        elif op_name == "Opening":
            result = ip.morphology_open(img, kernel_size, iterations)
        elif op_name == "Closing":
            result = ip.morphology_close(img, kernel_size, iterations)
        elif op_name == "Gradient":
            result = ip.morphology_gradient(img, kernel_size)
        elif op_name == "Top Hat":
            result = ip.morphology_tophat(img, kernel_size)
        elif op_name == "Black Hat":
            result = ip.morphology_blackhat(img, kernel_size)
        else:
            result = img.copy()

        self._processed_bgr = result
        self._show_processed()
        self.statusBar().showMessage(
            f"{op_name} applied (kernel={kernel_size}, iter={iterations})"
        )

    # ── Internal helpers ──────────────────────────────────────────── #

    def _show_processed(self):
        rgb = cv2.cvtColor(self._processed_bgr, cv2.COLOR_BGR2RGB)
        pm = self._numpy_to_pixmap(rgb)
        self.img_processed.set_pixmap(pm)
        self.btn_save.setEnabled(True)

    @staticmethod
    def _numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()