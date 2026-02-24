import os
import sys
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox, QLineEdit, QGroupBox
)

# 1) 改成你的 pipeline 文件名（确保 run_once 已加入）
from zhanlan.ivfflat.search_ivfflat_patch_global_stripe_sort_修改测试 import run_once


def cv_bgr_to_qpixmap(img_bgr: np.ndarray, max_w=520, max_h=520) -> QPixmap:
    if img_bgr is None or img_bgr.size == 0:
        return QPixmap()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # resize for display (keep aspect)
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img_rgb.shape[:2]

    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def read_image_unicode(path: str):
    # 用 cv2.imdecode 支持中文路径
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


class RetrievalUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zhanlan Hybrid Retrieval UI (PyQt5)")
        self.resize(1700, 720)

        self.query_path = None
        self.crop_path = None
        self.result_path = None

        # ---- controls
        self.btn_open = QPushButton("打开图片")
        self.btn_run = QPushButton("运行检索")
        self.btn_run.setEnabled(False)

        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("输出目录（可选，不填则用 pipeline 的 OUT_DIR）")

        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)

        # ---- image views
        self.lbl_query = QLabel("原图")
        self.lbl_crop = QLabel("切割后图")
        self.lbl_result = QLabel("结果图 (result_grid1.png)")

        for lb in (self.lbl_query, self.lbl_crop, self.lbl_result):
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet("border: 1px solid #999; background: #111; color: #ddd;")
            lb.setMinimumSize(520, 520)

        # ---- layout
        top_box = QGroupBox("操作")
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.btn_run)
        top_layout.addWidget(QLabel("输出目录:"))
        top_layout.addWidget(self.out_dir_edit, 1)
        top_box.setLayout(top_layout)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.wrap_with_title("原图", self.lbl_query))
        img_layout.addWidget(self.wrap_with_title("切割后图", self.lbl_crop))
        img_layout.addWidget(self.wrap_with_title("结果图", self.lbl_result))

        main_layout = QVBoxLayout()
        main_layout.addWidget(top_box)
        main_layout.addLayout(img_layout, 1)
        self.setLayout(main_layout)

    def wrap_with_title(self, title: str, widget: QWidget) -> QWidget:
        box = QGroupBox(title)
        lay = QVBoxLayout()
        lay.addWidget(widget)
        box.setLayout(lay)
        return box

    def set_label_image(self, label: QLabel, img_path: str):
        img = read_image_unicode(img_path)
        if img is None:
            label.setText("读取失败")
            return
        label.setPixmap(cv_bgr_to_qpixmap(img))

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        self.query_path = path
        self.crop_path = None
        self.result_path = None

        self.set_label_image(self.lbl_query, self.query_path)
        self.lbl_crop.setText("切割后图")
        self.lbl_result.setText("结果图 (result_grid1.png)")
        self.btn_run.setEnabled(True)

    def on_run(self):
        if not self.query_path or not os.path.exists(self.query_path):
            QMessageBox.warning(self, "提示", "请先打开一张图片。")
            return

        out_dir = self.out_dir_edit.text().strip() or None

        try:
            self.btn_run.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)

            ret = run_once(self.query_path, out_dir=out_dir)

            self.crop_path = ret.get("crop_path", None)
            self.result_path = ret.get("result_path", None)

            if self.crop_path and os.path.exists(self.crop_path):
                self.set_label_image(self.lbl_crop, self.crop_path)
            else:
                self.lbl_crop.setText("未生成 crop.png")

            if self.result_path and os.path.exists(self.result_path):
                self.set_label_image(self.lbl_result, self.result_path)
            else:
                self.lbl_result.setText("未生成 result_grid1.png")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "运行失败", f"{e}\n\n{tb}")
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RetrievalUI()
    w.show()
    sys.exit(app.exec_())