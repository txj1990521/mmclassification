#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import shutil

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# 注意：run_retrieval 现在返回 (result_path, top_items)
from retrieval_pipeline import run_retrieval


def imread_unicode(path: str):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def bgr_to_pg_image(bgr: np.ndarray):
    if bgr is None:
        return None
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Worker(QtCore.QThread):
    done = QtCore.pyqtSignal(str, object)  # result_path, top_items(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, query_path: str, out_dir: str, topk: int):
        super().__init__()
        self.query_path = query_path
        self.out_dir = out_dir
        self.topk = topk

    def run(self):
        try:
            result_path, top_items = run_retrieval(self.query_path, self.out_dir, self.topk)
            if not os.path.exists(result_path):
                raise RuntimeError(f"Result not found: {result_path}")
            self.done.emit(result_path, top_items)
        except Exception as e:
            self.error.emit(str(e))


class ImagePanel(pg.GraphicsLayoutWidget):
    def __init__(self, title: str):
        super().__init__()
        self.setWindowTitle(title)

        self.view = self.addViewBox(row=0, col=0)
        self.view.setAspectLocked(True)
        self.view.invertY(True)   # 关键：图像坐标系(0,0)在左上
        # self.view.invertX(True)

        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

        self._current_path = None

    def set_image_bgr(self, bgr: np.ndarray, path: str = None):
        rgb = bgr_to_pg_image(bgr)
        self._current_path = path
        if rgb is None:
            self.img_item.clear()
            return
        self.img_item.setImage(rgb, autoLevels=True)
        self.view.autoRange()

    def current_path(self):
        return self._current_path



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Retrieval GUI (pyqtgraph + list)")
        self.resize(1400, 760)

        self.query_path = None
        self.result_grid_path = None
        self.top_items = []  # [{"path":..., "score":..., "geom":...}, ...]

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- 控制区
        ctrl = QtWidgets.QHBoxLayout()
        root.addLayout(ctrl)

        self.btn_open = QtWidgets.QPushButton("选择图片")
        self.btn_run = QtWidgets.QPushButton("开始识别/检索")
        self.btn_save = QtWidgets.QPushButton("保存右侧当前预览")
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)

        ctrl.addWidget(self.btn_open)
        ctrl.addWidget(self.btn_run)
        ctrl.addWidget(self.btn_save)

        ctrl.addSpacing(16)
        ctrl.addWidget(QtWidgets.QLabel("TOPK:"))
        self.spin_topk = QtWidgets.QSpinBox()
        self.spin_topk.setRange(1, 100)
        self.spin_topk.setValue(12)
        ctrl.addWidget(self.spin_topk)

        ctrl.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("就绪")
        ctrl.addWidget(self.lbl_status)

        # --- 下方内容区：左(原图) + 右(列表+预览)
        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, 1)

        self.panel_query = ImagePanel("原图")
        content.addWidget(self.panel_query, 1)

        # 右侧：列表 + 预览
        right = QtWidgets.QVBoxLayout()
        content.addLayout(right, 1)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMinimumWidth(360)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        right.addWidget(self.list_widget, 0)

        self.panel_preview = ImagePanel("右侧预览")
        right.addWidget(self.panel_preview, 1)

        # --- 信号
        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save_preview)
        self.list_widget.currentRowChanged.connect(self.on_list_change)

        self.worker = None
        self.out_dir = os.path.abspath("./search_vis")
        os.makedirs(self.out_dir, exist_ok=True)

    def set_status(self, s: str):
        self.lbl_status.setText(s)

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 Query 图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)"
        )
        if not path:
            return

        bgr = imread_unicode(path)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "错误", "图片读取失败（路径/编码可能有问题）")
            return

        self.query_path = path
        self.panel_query.set_image_bgr(bgr, path=path)

        # 清空右侧
        self.top_items = []
        self.result_grid_path = None
        self.list_widget.clear()
        self.panel_preview.set_image_bgr(None, path=None)

        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.set_status(f"已选择：{os.path.basename(path)}")

    def on_run(self):
        if not self.query_path:
            return

        self.btn_open.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.list_widget.setEnabled(False)
        self.list_widget.clear()
        self.panel_preview.set_image_bgr(None, path=None)

        self.set_status("检索中...")

        topk = int(self.spin_topk.value())
        self.worker = Worker(self.query_path, self.out_dir, topk)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_done(self, result_grid_path: str, top_items: object):
        self.result_grid_path = result_grid_path
        self.top_items = list(top_items) if top_items is not None else []

        # 1) 填充列表
        self.list_widget.clear()
        # 加一个“网格预览”项放最上面
        self.list_widget.addItem("【GRID】result_grid.png")

        for idx, it in enumerate(self.top_items, start=1):
            p = it.get("path", "")
            score = it.get("score", 0.0)
            geom = it.get("geom", 0.0)
            name = os.path.basename(p)
            self.list_widget.addItem(f"{idx:02d}  score={score:.3f}  geom={geom:.3f}  {name}")

        # 2) 默认显示网格图（选中第0项）
        self.list_widget.setCurrentRow(0)

        self.btn_open.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.list_widget.setEnabled(True)
        self.set_status("完成")

    def on_error(self, msg: str):
        self.btn_open.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.list_widget.setEnabled(True)
        self.btn_save.setEnabled(bool(self.panel_preview.current_path()))
        self.set_status("失败")
        QtWidgets.QMessageBox.critical(self, "运行失败", msg)

    def on_list_change(self, row: int):
        if row < 0:
            return

        # row == 0 -> 显示 result_grid.png
        if row == 0:
            if self.result_grid_path and os.path.exists(self.result_grid_path):
                bgr = imread_unicode(self.result_grid_path)
                self.panel_preview.set_image_bgr(bgr, path=self.result_grid_path)
                self.btn_save.setEnabled(True)
            else:
                self.panel_preview.set_image_bgr(None, path=None)
                self.btn_save.setEnabled(False)
            return

        # row >= 1 -> 显示 top_items[row-1]
        idx = row - 1
        if idx >= len(self.top_items):
            return

        p = self.top_items[idx].get("path", "")
        if p and os.path.exists(p):
            bgr = imread_unicode(p)
            self.panel_preview.set_image_bgr(bgr, path=p)
            self.btn_save.setEnabled(True)
        else:
            self.panel_preview.set_image_bgr(None, path=None)
            self.btn_save.setEnabled(False)

    def on_save_preview(self):
        cur = self.panel_preview.current_path()
        if not cur or not os.path.exists(cur):
            QtWidgets.QMessageBox.information(self, "提示", "当前没有可保存的预览图")
            return

        default_name = "preview_saved.png"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存右侧当前预览",
            os.path.join(os.path.dirname(cur), default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not save_path:
            return

        try:
            # 直接复制文件（最稳，不引入颜色/压缩差异）
            shutil.copyfile(cur, save_path)
            self.set_status(f"已保存：{os.path.basename(save_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "保存失败", str(e))


def main():
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
