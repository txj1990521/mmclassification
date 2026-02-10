# app_pyqt.py
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from hybrid_wrapper import run_pipeline


def read_image_unicode(path: str) -> np.ndarray:
    """cv2.imread 不稳定时，兼容中文路径"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class Worker(QtCore.QThread):
    sig_done = QtCore.pyqtSignal(object, str, object, object)  # qimg_bgr, grid_path, top, debug
    sig_err = QtCore.pyqtSignal(str)

    def __init__(self, query_path: str, out_dir: str, topk: int, parent=None):
        super().__init__(parent)
        self.query_path = query_path
        self.out_dir = out_dir
        self.topk = topk

    def run(self):
        try:
            qimg, grid_path, top, debug = run_pipeline(
                self.query_path,
                out_dir=self.out_dir,
                topk=self.topk
            )
            self.sig_done.emit(qimg, grid_path, top, debug)
        except Exception as e:
            self.sig_err.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zhanlan Hybrid Retrieval UI (PyQt5 + pyqtgraph)")
        self.resize(1450, 900)

        self.query_path = None
        self.worker = None

        # pyqtgraph 配置：非常重要（图像轴顺序）
        pg.setConfigOptions(imageAxisOrder="row-major")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # 控件：按钮 / 参数
        self.btn_pick = QtWidgets.QPushButton("选择图像")
        self.btn_run = QtWidgets.QPushButton("开始检索")
        self.btn_run.setEnabled(False)

        self.edt_out = QtWidgets.QLineEdit(os.path.abspath("./search_vis_ui"))
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 200)
        self.spin_topk.setValue(12)

        self.lbl_status = QtWidgets.QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#333;")

        # 左右显示：query / result_grid
        self.view_query = pg.ImageView()
        self.view_result = pg.ImageView()
        self.view_query.ui.roiBtn.hide(); self.view_query.ui.menuBtn.hide()
        self.view_result.ui.roiBtn.hide(); self.view_result.ui.menuBtn.hide()

        # top table
        self.tbl_top = QtWidgets.QTableWidget(0, 2)
        self.tbl_top.setHorizontalHeaderLabels(["img_id", "score"])
        self.tbl_top.horizontalHeader().setStretchLastSection(True)
        self.tbl_top.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_top.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        # 顶部栏
        topbar = QtWidgets.QHBoxLayout()
        topbar.addWidget(self.btn_pick)
        topbar.addWidget(self.btn_run)
        topbar.addSpacing(12)
        topbar.addWidget(QtWidgets.QLabel("输出目录:"))
        topbar.addWidget(self.edt_out, 1)
        topbar.addWidget(QtWidgets.QLabel("TOPK:"))
        topbar.addWidget(self.spin_topk)
        topbar.addStretch(1)

        # 主区分割
        splitter = QtWidgets.QSplitter()

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("Query（裁剪后显示）"))
        left_layout.addWidget(self.view_query, 1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("结果拼图（result_grid1.png）"))
        right_layout.addWidget(self.view_result, 2)
        right_layout.addWidget(QtWidgets.QLabel("Top 列表"))
        right_layout.addWidget(self.tbl_top, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([650, 800])

        root = QtWidgets.QVBoxLayout(central)
        root.addLayout(topbar)
        root.addWidget(splitter, 1)
        root.addWidget(self.lbl_status)

        # 信号
        self.btn_pick.clicked.connect(self.on_pick)
        self.btn_run.clicked.connect(self.on_run)

    def on_pick(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 Query 图像",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        self.query_path = path
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"已选择: {path}")

        # 预览原图
        img = read_image_unicode(path)
        if img is not None:
            self.view_query.setImage(bgr_to_rgb(img), autoLevels=True)

    def on_run(self):
        if not self.query_path:
            return

        out_dir = self.edt_out.text().strip()
        if not out_dir:
            out_dir = os.path.abspath("./search_vis_ui")
            self.edt_out.setText(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        topk = int(self.spin_topk.value())

        self.btn_run.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.lbl_status.setText("运行中...（首次加载模型会慢一点）")

        self.worker = Worker(self.query_path, out_dir, topk, self)
        self.worker.sig_done.connect(self.on_done)
        self.worker.sig_err.connect(self.on_err)
        self.worker.start()

    def on_done(self, qimg_bgr, grid_path, top, debug):
        self.btn_run.setEnabled(True)
        self.btn_pick.setEnabled(True)

        # 显示裁剪后的 query
        if qimg_bgr is not None:
            self.view_query.setImage(bgr_to_rgb(qimg_bgr), autoLevels=True)

        # 显示拼图
        if grid_path and os.path.exists(grid_path):
            grid_bgr = read_image_unicode(grid_path)
            if grid_bgr is not None:
                self.view_result.setImage(bgr_to_rgb(grid_bgr), autoLevels=True)

        # top table
        self.tbl_top.setRowCount(0)
        for r, (img_id, score) in enumerate(top):
            self.tbl_top.insertRow(r)
            self.tbl_top.setItem(r, 0, QtWidgets.QTableWidgetItem(str(int(img_id))))
            self.tbl_top.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(score):.6f}"))

        # status
        msg = f"DONE: {os.path.basename(grid_path)}"
        if isinstance(debug, dict):
            msg += f" | stripe={debug.get('is_stripe')} n_qpatch={debug.get('n_qpatch')} fallback={debug.get('fallback')}"
            if "w_g" in debug:
                msg += f" w_g={debug.get('w_g'):.3f} w_p={debug.get('w_p'):.3f}"
        self.lbl_status.setText(msg)

    def on_err(self, err: str):
        self.btn_run.setEnabled(True)
        self.btn_pick.setEnabled(True)
        self.lbl_status.setText("ERROR: " + err)
        QtWidgets.QMessageBox.critical(self, "运行失败", err)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
