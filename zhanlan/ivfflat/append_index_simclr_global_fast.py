# index_manager_gui.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import faiss

from PyQt5 import QtCore, QtWidgets


# =========================
# 你只改这里（默认值）
# =========================
PYTHON_EXE = r"D:\ProgramData\miniconda3\envs\mmlab_stable\python.exe"

CLIP_APPEND_SCRIPT = r"D:\zhanlanProject\openai_search\append_index_hybrid_clip_folder_ivf_pq_fast.py"
SIMCLR_TAIL_APPEND_SCRIPT = r"D:\zhanlanProject\mmpretrain\zhanlan\ivfflat\append_simclr_tail_from_images_meta.py"

CLIP_OUT_DIR_DEFAULT = r"D:\zhanlanProject\openai_search\outputs_hybrid_folder_big"
SIMCLR_DIR_DEFAULT = r"D:\zhanlan\faiss_database_simclr_aligned"
IMAGES_META_DEFAULT = os.path.join(CLIP_OUT_DIR_DEFAULT, "images_meta.json")

# SimCLR model defaults
SIMCLR_CFG_DEFAULT = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
SIMCLR_CKPT_DEFAULT = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

# default simclr files
SIMCLR_INDEX_NAME_DEFAULT = "simclr_global.index"
SIMCLR_VEC2IMG_NAME_DEFAULT = "simclr_vec_to_imgid.npy"


# =========================
# Data
# =========================
@dataclass
class RunPlan:
    enable_clip: bool
    enable_simclr: bool
    new_roots: List[str]
    dry_run: bool


# =========================
# Worker: running subprocess
# =========================
class ProcWorker(QtCore.QThread):
    sig_log = QtCore.pyqtSignal(str)
    sig_done = QtCore.pyqtSignal(int)

    def __init__(self, cmd: List[str], cwd: Optional[str] = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.p: Optional[subprocess.Popen] = None

    def run(self):
        try:
            self.sig_log.emit("[CMD] " + " ".join(self.cmd))
            self.p = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            assert self.p.stdout is not None
            for line in self.p.stdout:
                self.sig_log.emit(line.rstrip("\n"))
            ret = self.p.wait()
            self.sig_done.emit(ret)
        except Exception as e:
            self.sig_log.emit("[ERROR] " + repr(e))
            self.sig_done.emit(-1)

    def kill_tree_windows(self):
        if self.p is None:
            return
        try:
            pid = self.p.pid
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"])
        except Exception:
            pass


# =========================
# Worker: alignment status (background)
# =========================
class StatusWorker(QtCore.QThread):
    sig_status = QtCore.pyqtSignal(dict)

    def __init__(self, clip_out_dir: str, images_meta_json: str, simclr_dir: str,
                 simclr_index_name_or_path: str, simclr_vec2img_name_or_path: str):
        super().__init__()
        self.clip_out_dir = clip_out_dir
        self.images_meta_json = images_meta_json
        self.simclr_dir = simclr_dir
        self.simclr_index = simclr_index_name_or_path
        self.simclr_vec2img = simclr_vec2img_name_or_path

    @staticmethod
    def _smart_join(base_dir: str, maybe_path: str) -> str:
        if not maybe_path:
            return ""
        if os.path.isabs(maybe_path) or (":" in maybe_path):
            return maybe_path
        return os.path.join(base_dir, maybe_path)

    @staticmethod
    def _safe_len_images_meta(path: str) -> Tuple[Optional[int], Optional[str]]:
        try:
            if not path or (not os.path.exists(path)):
                return None, "images_meta.json not found"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return None, "images_meta.json is not list"
            return len(data), None
        except Exception as e:
            return None, f"images_meta read fail: {repr(e)}"

    @staticmethod
    def _safe_len_npy(path: str) -> Tuple[Optional[int], Optional[str]]:
        try:
            if not path or (not os.path.exists(path)):
                return None, "vec2img npy not found"
            arr = np.load(path, allow_pickle=False)
            return int(arr.shape[0]), None
        except Exception as e:
            return None, f"npy read fail: {repr(e)}"

    @staticmethod
    def _safe_faiss_ntotal(path: str) -> Tuple[Optional[int], Optional[str]]:
        try:
            if not path or (not os.path.exists(path)):
                return None, "index not found"
            idx = faiss.read_index(path)
            return int(idx.ntotal), None
        except Exception as e:
            return None, f"faiss read fail: {repr(e)}"

    def run(self):
        out: Dict[str, Any] = {
            "images_meta_len": None,
            "clip_global_ntotal": None,
            "clip_patch_ntotal": None,
            "simclr_ntotal": None,
            "simclr_vec2img_len": None,
            "aligned": None,
            "errors": [],
            "paths": {},
        }

        # paths
        images_meta_path = self.images_meta_json
        g_index_path = os.path.join(self.clip_out_dir, "global.index")
        p_index_path = os.path.join(self.clip_out_dir, "patch.index")

        simclr_index_path = self._smart_join(self.simclr_dir, self.simclr_index)
        simclr_vec2img_path = self._smart_join(self.simclr_dir, self.simclr_vec2img)

        out["paths"] = {
            "images_meta": images_meta_path,
            "clip_global_index": g_index_path,
            "clip_patch_index": p_index_path,
            "simclr_index": simclr_index_path,
            "simclr_vec2img": simclr_vec2img_path,
        }

        # read images_meta len
        n_meta, err = self._safe_len_images_meta(images_meta_path)
        out["images_meta_len"] = n_meta
        if err:
            out["errors"].append(err)

        # clip ntotal
        ng, errg = self._safe_faiss_ntotal(g_index_path)
        out["clip_global_ntotal"] = ng
        if errg:
            out["errors"].append("CLIP global: " + errg)

        npatch, errp = self._safe_faiss_ntotal(p_index_path)
        out["clip_patch_ntotal"] = npatch
        if errp:
            out["errors"].append("CLIP patch: " + errp)

        # simclr ntotal + vec2img
        ns, errs = self._safe_faiss_ntotal(simclr_index_path)
        out["simclr_ntotal"] = ns
        if errs:
            out["errors"].append("SimCLR index: " + errs)

        nmap, errm = self._safe_len_npy(simclr_vec2img_path)
        out["simclr_vec2img_len"] = nmap
        if errm:
            out["errors"].append("SimCLR vec2img: " + errm)

        # alignment
        if (n_meta is not None) and (nmap is not None):
            out["aligned"] = bool(int(n_meta) == int(nmap))
        else:
            out["aligned"] = None

        self.sig_status.emit(out)


# =========================
# GUI
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("增量建库管理器 (CLIP + SimCLR, 方案2/模式A尾巴对齐)")
        self.resize(1150, 900)

        self.worker: Optional[ProcWorker] = None
        self.status_worker: Optional[StatusWorker] = None
        self.queue: List[List[str]] = []
        self.log_file_path: Optional[str] = None

        self._status_debounce = QtCore.QTimer()
        self._status_debounce.setSingleShot(True)
        self._status_debounce.timeout.connect(self.refresh_status)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # -------- Alignment status --------
        grp_status = QtWidgets.QGroupBox("对齐状态（images_meta / CLIP / SimCLR）")
        layout.addWidget(grp_status)
        gs = QtWidgets.QGridLayout(grp_status)

        self.lbl_meta = QtWidgets.QLabel("images_meta 长度: -")
        self.lbl_clip_g = QtWidgets.QLabel("CLIP GLOBAL ntotal: -")
        self.lbl_clip_p = QtWidgets.QLabel("CLIP PATCH  ntotal: -")
        self.lbl_sim_ntotal = QtWidgets.QLabel("SimCLR ntotal: -")
        self.lbl_sim_map = QtWidgets.QLabel("SimCLR vec2img_len: -")
        self.lbl_align = QtWidgets.QLabel("对齐: -")
        self.lbl_align.setStyleSheet("font-weight: bold;")

        self.btn_refresh = QtWidgets.QPushButton("刷新状态")
        self.btn_refresh.clicked.connect(self.refresh_status)

        self.txt_status_err = QtWidgets.QPlainTextEdit()
        self.txt_status_err.setReadOnly(True)
        self.txt_status_err.setMaximumHeight(80)

        gs.addWidget(self.lbl_meta, 0, 0)
        gs.addWidget(self.lbl_clip_g, 0, 1)
        gs.addWidget(self.lbl_clip_p, 0, 2)
        gs.addWidget(self.lbl_sim_ntotal, 1, 0)
        gs.addWidget(self.lbl_sim_map, 1, 1)
        gs.addWidget(self.lbl_align, 1, 2)
        gs.addWidget(self.btn_refresh, 0, 3, 2, 1)
        gs.addWidget(self.txt_status_err, 2, 0, 1, 4)

        # -------- Roots selection --------
        grp_roots = QtWidgets.QGroupBox("新增图片目录 (可多选，按顺序追加)")
        layout.addWidget(grp_roots)
        v = QtWidgets.QVBoxLayout(grp_roots)

        self.lst_roots = QtWidgets.QListWidget()
        self.lst_roots.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        v.addWidget(self.lst_roots, 1)

        row = QtWidgets.QHBoxLayout()
        v.addLayout(row)
        self.btn_add_root = QtWidgets.QPushButton("添加目录")
        self.btn_del_root = QtWidgets.QPushButton("删除选中")
        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_up = QtWidgets.QPushButton("上移")
        self.btn_down = QtWidgets.QPushButton("下移")
        row.addWidget(self.btn_add_root)
        row.addWidget(self.btn_del_root)
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_up)
        row.addWidget(self.btn_down)
        row.addStretch(1)

        # -------- Options tabs --------
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # ===== Tab: Common =====
        tab_common = QtWidgets.QWidget()
        tabs.addTab(tab_common, "通用")
        g0 = QtWidgets.QGridLayout(tab_common)
        r = 0

        self.chk_dry = QtWidgets.QCheckBox("Dry-run：只打印命令，不执行")
        g0.addWidget(self.chk_dry, r, 0, 1, 2)
        r += 1

        g0.addWidget(QtWidgets.QLabel("Python EXE:"), r, 0)
        self.ed_py = QtWidgets.QLineEdit(PYTHON_EXE)
        btn_py = QtWidgets.QPushButton("选择")
        g0.addWidget(self.ed_py, r, 1)
        g0.addWidget(btn_py, r, 2)
        r += 1

        g0.addWidget(QtWidgets.QLabel("CLIP Append Script:"), r, 0)
        self.ed_clip_script = QtWidgets.QLineEdit(CLIP_APPEND_SCRIPT)
        btn_clip_script = QtWidgets.QPushButton("选择")
        g0.addWidget(self.ed_clip_script, r, 1)
        g0.addWidget(btn_clip_script, r, 2)
        r += 1

        g0.addWidget(QtWidgets.QLabel("SimCLR Tail Script:"), r, 0)
        self.ed_sim_script = QtWidgets.QLineEdit(SIMCLR_TAIL_APPEND_SCRIPT)
        btn_sim_script = QtWidgets.QPushButton("选择")
        g0.addWidget(self.ed_sim_script, r, 1)
        g0.addWidget(btn_sim_script, r, 2)
        r += 1

        g0.addWidget(QtWidgets.QLabel("保存日志到文件:"), r, 0)
        self.ed_log_file = QtWidgets.QLineEdit("")
        btn_log = QtWidgets.QPushButton("选择")
        g0.addWidget(self.ed_log_file, r, 1)
        g0.addWidget(btn_log, r, 2)
        r += 1

        g0.setColumnStretch(1, 1)

        # ===== Tab: CLIP =====
        tab_clip = QtWidgets.QWidget()
        tabs.addTab(tab_clip, "CLIP（global/patch）")
        g1 = QtWidgets.QGridLayout(tab_clip)
        rr = 0

        self.chk_clip = QtWidgets.QCheckBox("更新 CLIP 库（更新 images_meta.json + global/patch index + maps）")
        self.chk_clip.setChecked(True)
        g1.addWidget(self.chk_clip, rr, 0, 1, 3)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("OUT_DIR:"), rr, 0)
        self.ed_clip_out = QtWidgets.QLineEdit(CLIP_OUT_DIR_DEFAULT)
        btn_out = QtWidgets.QPushButton("选择")
        g1.addWidget(self.ed_clip_out, rr, 1)
        g1.addWidget(btn_out, rr, 2)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("device:"), rr, 0)
        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.addItems(["auto", "cpu", "cuda"])
        self.cb_device.setCurrentText("auto")
        g1.addWidget(self.cb_device, rr, 1)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("dedup_by:"), rr, 0)
        self.cb_dedup = QtWidgets.QComboBox()
        self.cb_dedup.addItems(["key", "abs", "both"])
        self.cb_dedup.setCurrentText("key")
        g1.addWidget(self.cb_dedup, rr, 1)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("max_new_images (0=不限):"), rr, 0)
        self.sp_max_new = QtWidgets.QSpinBox()
        self.sp_max_new.setRange(0, 10_000_000)
        self.sp_max_new.setValue(0)
        g1.addWidget(self.sp_max_new, rr, 1)
        rr += 1

        self.chk_patch_enable = QtWidgets.QCheckBox("PATCH_ENABLE（不勾选则只追加 global）")
        self.chk_patch_enable.setChecked(True)
        g1.addWidget(self.chk_patch_enable, rr, 0, 1, 3)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("PATCH_BATCH:"), rr, 0)
        self.sp_patch_batch = QtWidgets.QSpinBox()
        self.sp_patch_batch.setRange(1, 4096)
        self.sp_patch_batch.setValue(256)
        g1.addWidget(self.sp_patch_batch, rr, 1)
        rr += 1

        g1.addWidget(QtWidgets.QLabel("PATCH_RESIZE_LONG (0=不缩放):"), rr, 0)
        self.sp_patch_resize = QtWidgets.QSpinBox()
        self.sp_patch_resize.setRange(0, 10000)
        self.sp_patch_resize.setValue(1600)
        g1.addWidget(self.sp_patch_resize, rr, 1)
        rr += 1

        self.chk_5crop = QtWidgets.QCheckBox("GLOBAL_USE_5CROP")
        self.chk_5crop.setChecked(True)
        g1.addWidget(self.chk_5crop, rr, 0, 1, 1)

        g1.addWidget(QtWidgets.QLabel("GLOBAL_CROP_RATIO:"), rr, 1)
        self.dsp_crop_ratio = QtWidgets.QDoubleSpinBox()
        self.dsp_crop_ratio.setRange(0.1, 1.0)
        self.dsp_crop_ratio.setSingleStep(0.05)
        self.dsp_crop_ratio.setValue(0.8)
        g1.addWidget(self.dsp_crop_ratio, rr, 2)
        rr += 1

        g1.setColumnStretch(1, 1)

        # ===== Tab: SimCLR =====
        tab_sim = QtWidgets.QWidget()
        tabs.addTab(tab_sim, "SimCLR（模式A尾巴对齐）")
        g2 = QtWidgets.QGridLayout(tab_sim)
        rr = 0

        self.chk_simclr = QtWidgets.QCheckBox("更新 SimCLR 库（模式A：不改 images_meta，只补尾巴）")
        self.chk_simclr.setChecked(True)
        g2.addWidget(self.chk_simclr, rr, 0, 1, 3)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("SimCLR OUT_DIR:"), rr, 0)
        self.ed_simclr_dir = QtWidgets.QLineEdit(SIMCLR_DIR_DEFAULT)
        btn_simdir = QtWidgets.QPushButton("选择")
        g2.addWidget(self.ed_simclr_dir, rr, 1)
        g2.addWidget(btn_simdir, rr, 2)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("SimCLR INDEX 文件名/路径:"), rr, 0)
        self.ed_sim_index = QtWidgets.QLineEdit(SIMCLR_INDEX_NAME_DEFAULT)
        g2.addWidget(self.ed_sim_index, rr, 1)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("SimCLR VEC2IMG 文件名/路径:"), rr, 0)
        self.ed_sim_vec2img = QtWidgets.QLineEdit(SIMCLR_VEC2IMG_NAME_DEFAULT)
        g2.addWidget(self.ed_sim_vec2img, rr, 1)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("images_meta.json:"), rr, 0)
        self.ed_meta = QtWidgets.QLineEdit(IMAGES_META_DEFAULT)
        btn_meta = QtWidgets.QPushButton("选择")
        g2.addWidget(self.ed_meta, rr, 1)
        g2.addWidget(btn_meta, rr, 2)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("CFG_PATH:"), rr, 0)
        self.ed_sim_cfg = QtWidgets.QLineEdit(SIMCLR_CFG_DEFAULT)
        btn_cfg = QtWidgets.QPushButton("选择")
        g2.addWidget(self.ed_sim_cfg, rr, 1)
        g2.addWidget(btn_cfg, rr, 2)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("CKPT_PATH:"), rr, 0)
        self.ed_sim_ckpt = QtWidgets.QLineEdit(SIMCLR_CKPT_DEFAULT)
        btn_ckpt = QtWidgets.QPushButton("选择")
        g2.addWidget(self.ed_sim_ckpt, rr, 1)
        g2.addWidget(btn_ckpt, rr, 2)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("device:"), rr, 0)
        self.cb_sim_device = QtWidgets.QComboBox()
        self.cb_sim_device.addItems(["auto", "cpu", "cuda"])
        self.cb_sim_device.setCurrentText("auto")
        g2.addWidget(self.cb_sim_device, rr, 1)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("BATCH:"), rr, 0)
        self.sp_sim_batch = QtWidgets.QSpinBox()
        self.sp_sim_batch.setRange(1, 8192)
        self.sp_sim_batch.setValue(256)
        g2.addWidget(self.sp_sim_batch, rr, 1)
        rr += 1

        g2.addWidget(QtWidgets.QLabel("INPUT_SIZE:"), rr, 0)
        self.sp_sim_in = QtWidgets.QSpinBox()
        self.sp_sim_in.setRange(64, 1024)
        self.sp_sim_in.setValue(224)
        g2.addWidget(self.sp_sim_in, rr, 1)
        rr += 1

        self.chk_strict = QtWidgets.QCheckBox("STRICT_ALIGN（强制对齐：遇到坏图直接失败）")
        self.chk_strict.setChecked(True)
        g2.addWidget(self.chk_strict, rr, 0, 1, 3)
        rr += 1

        g2.setColumnStretch(1, 1)

        # -------- Actions --------
        act = QtWidgets.QHBoxLayout()
        layout.addLayout(act)
        self.btn_run = QtWidgets.QPushButton("开始执行")
        self.btn_stop = QtWidgets.QPushButton("停止(强制)")
        self.btn_stop.setEnabled(False)
        act.addWidget(self.btn_run)
        act.addWidget(self.btn_stop)
        act.addStretch(1)

        # -------- Log --------
        grp_log = QtWidgets.QGroupBox("日志输出")
        layout.addWidget(grp_log, 1)
        lv = QtWidgets.QVBoxLayout(grp_log)
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        lv.addWidget(self.txt_log, 1)

        # signals: roots ops
        self.btn_add_root.clicked.connect(self.on_add_root)
        self.btn_del_root.clicked.connect(self.on_del_root)
        self.btn_clear.clicked.connect(self.on_clear_roots)
        self.btn_up.clicked.connect(self.on_up)
        self.btn_down.clicked.connect(self.on_down)

        # run/stop
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop.clicked.connect(self.on_stop)

        # file pickers
        btn_py.clicked.connect(lambda: self.pick_file(self.ed_py, "选择 python.exe", "Executable (*.exe);;All (*)"))
        btn_clip_script.clicked.connect(lambda: self.pick_file(self.ed_clip_script, "选择 CLIP 脚本", "Python (*.py);;All (*)"))
        btn_sim_script.clicked.connect(lambda: self.pick_file(self.ed_sim_script, "选择 SimCLR 脚本", "Python (*.py);;All (*)"))
        btn_out.clicked.connect(lambda: self.pick_dir(self.ed_clip_out, "选择 CLIP OUT_DIR"))
        btn_simdir.clicked.connect(lambda: self.pick_dir(self.ed_simclr_dir, "选择 SimCLR OUT_DIR"))
        btn_meta.clicked.connect(lambda: self.pick_file(self.ed_meta, "选择 images_meta.json", "JSON (*.json);;All (*)"))
        btn_cfg.clicked.connect(lambda: self.pick_file(self.ed_sim_cfg, "选择 SimCLR cfg", "Python (*.py);;All (*)"))
        btn_ckpt.clicked.connect(lambda: self.pick_file(self.ed_sim_ckpt, "选择 SimCLR ckpt", "Checkpoint (*.pth *.pt);;All (*)"))
        btn_log.clicked.connect(self.on_pick_logfile)

        # auto refresh status when key paths change (debounced)
        for w in [self.ed_clip_out, self.ed_meta, self.ed_simclr_dir, self.ed_sim_index, self.ed_sim_vec2img]:
            w.textChanged.connect(lambda _=None: self._status_debounce.start(350))

        # initial refresh
        self.refresh_status()

    # ---------- helpers ----------
    def pick_dir(self, line: QtWidgets.QLineEdit, title: str):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, title)
        if d:
            line.setText(d)

    def pick_file(self, line: QtWidgets.QLineEdit, title: str, flt: str):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, "", flt)
        if p:
            line.setText(p)

    def on_pick_logfile(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "选择日志文件保存路径", "", "Text (*.log *.txt);;All (*)")
        if p:
            self.ed_log_file.setText(p)

    def log(self, s: str):
        self.txt_log.appendPlainText(s)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
            except Exception:
                pass

    # ---------- status ----------
    def refresh_status(self):
        # avoid piling multiple workers
        if self.status_worker is not None and self.status_worker.isRunning():
            return

        clip_out = self.ed_clip_out.text().strip()
        meta_json = self.ed_meta.text().strip()
        sim_dir = self.ed_simclr_dir.text().strip()
        sim_index = self.ed_sim_index.text().strip()
        sim_vec2img = self.ed_sim_vec2img.text().strip()

        self.lbl_align.setText("对齐: 读取中 ...")
        self.lbl_align.setStyleSheet("font-weight: bold; color: #444;")
        self.txt_status_err.setPlainText("")

        self.status_worker = StatusWorker(
            clip_out_dir=clip_out,
            images_meta_json=meta_json,
            simclr_dir=sim_dir,
            simclr_index_name_or_path=sim_index,
            simclr_vec2img_name_or_path=sim_vec2img
        )
        self.status_worker.sig_status.connect(self.on_status_ready)
        self.status_worker.start()

    def on_status_ready(self, st: dict):
        def fmt(v):
            return "-" if v is None else str(v)

        self.lbl_meta.setText(f"images_meta 长度: {fmt(st.get('images_meta_len'))}")
        self.lbl_clip_g.setText(f"CLIP GLOBAL ntotal: {fmt(st.get('clip_global_ntotal'))}")
        self.lbl_clip_p.setText(f"CLIP PATCH  ntotal: {fmt(st.get('clip_patch_ntotal'))}")
        self.lbl_sim_ntotal.setText(f"SimCLR ntotal: {fmt(st.get('simclr_ntotal'))}")
        self.lbl_sim_map.setText(f"SimCLR vec2img_len: {fmt(st.get('simclr_vec2img_len'))}")

        aligned = st.get("aligned", None)
        if aligned is True:
            self.lbl_align.setText("对齐: ✅ YES（SimCLR 已追平 images_meta）")
            self.lbl_align.setStyleSheet("font-weight: bold; color: #14833b;")
        elif aligned is False:
            self.lbl_align.setText("对齐: ❌ NO（需要 SimCLR catch-up）")
            self.lbl_align.setStyleSheet("font-weight: bold; color: #b01c1c;")
        else:
            self.lbl_align.setText("对齐: -（信息不完整）")
            self.lbl_align.setStyleSheet("font-weight: bold; color: #444;")

        errs = st.get("errors", []) or []
        if errs:
            self.txt_status_err.setPlainText("\n".join(errs))
        else:
            self.txt_status_err.setPlainText("OK")

        # cache latest status for pre-run checks
        self._latest_status = st

    # ---------- roots ops ----------
    def on_add_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择新增图片目录")
        if not d:
            return
        self.lst_roots.addItem(d)

    def on_del_root(self):
        for it in self.lst_roots.selectedItems():
            self.lst_roots.takeItem(self.lst_roots.row(it))

    def on_clear_roots(self):
        self.lst_roots.clear()

    def on_up(self):
        row = self.lst_roots.currentRow()
        if row <= 0:
            return
        item = self.lst_roots.takeItem(row)
        self.lst_roots.insertItem(row - 1, item)
        self.lst_roots.setCurrentRow(row - 1)

    def on_down(self):
        row = self.lst_roots.currentRow()
        if row < 0 or row >= self.lst_roots.count() - 1:
            return
        item = self.lst_roots.takeItem(row)
        self.lst_roots.insertItem(row + 1, item)
        self.lst_roots.setCurrentRow(row + 1)

    # ---------- plan ----------
    def collect_plan(self) -> RunPlan:
        roots = []
        for i in range(self.lst_roots.count()):
            roots.append(self.lst_roots.item(i).text())
        return RunPlan(
            enable_clip=self.chk_clip.isChecked(),
            enable_simclr=self.chk_simclr.isChecked(),
            new_roots=roots,
            dry_run=self.chk_dry.isChecked(),
        )

    def validate_basic(self, plan: RunPlan) -> bool:
        py = self.ed_py.text().strip()
        if not os.path.exists(py):
            QtWidgets.QMessageBox.critical(self, "错误", f"找不到 Python：\n{py}")
            return False
        if plan.enable_clip and not os.path.exists(self.ed_clip_script.text().strip()):
            QtWidgets.QMessageBox.critical(self, "错误", f"找不到 CLIP 脚本：\n{self.ed_clip_script.text().strip()}")
            return False
        if plan.enable_simclr and not os.path.exists(self.ed_sim_script.text().strip()):
            QtWidgets.QMessageBox.critical(self, "错误", f"找不到 SimCLR 脚本：\n{self.ed_sim_script.text().strip()}")
            return False
        if not plan.new_roots:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加至少一个新增图片目录")
            return False
        if not (plan.enable_clip or plan.enable_simclr):
            QtWidgets.QMessageBox.warning(self, "提示", "至少勾选一个：CLIP 或 SimCLR")
            return False
        for r in plan.new_roots:
            if not os.path.exists(r):
                QtWidgets.QMessageBox.critical(self, "错误", f"新增目录不存在：\n{r}")
                return False
        return True

    # ---------- build commands ----------
    def build_clip_cmd(self, plan: RunPlan) -> List[str]:
        py = self.ed_py.text().strip()
        script = self.ed_clip_script.text().strip()
        out_dir = self.ed_clip_out.text().strip()

        cmd = [py, script, "--out_dir", out_dir]

        for r in plan.new_roots:
            cmd += ["--new_root", r]

        cmd += ["--dedup_by", self.cb_dedup.currentText()]
        cmd += ["--device", self.cb_device.currentText()]

        maxn = int(self.sp_max_new.value())
        if maxn > 0:
            cmd += ["--max_new_images", str(maxn)]

        cmd += ["--patch_enable", "1" if self.chk_patch_enable.isChecked() else "0"]
        cmd += ["--patch_batch", str(int(self.sp_patch_batch.value()))]
        cmd += ["--patch_resize_long", str(int(self.sp_patch_resize.value()))]

        cmd += ["--global_use_5crop", "1" if self.chk_5crop.isChecked() else "0"]
        cmd += ["--global_crop_ratio", f"{float(self.dsp_crop_ratio.value()):.4f}"]
        return cmd

    def build_simclr_cmd(self) -> List[str]:
        py = self.ed_py.text().strip()
        script = self.ed_sim_script.text().strip()

        cmd = [
            py, script,
            "--images_meta", self.ed_meta.text().strip(),
            "--simclr_out_dir", self.ed_simclr_dir.text().strip(),
            "--simclr_index", self.ed_sim_index.text().strip(),
            "--simclr_vec2img", self.ed_sim_vec2img.text().strip(),
            "--cfg_path", self.ed_sim_cfg.text().strip(),
            "--ckpt_path", self.ed_sim_ckpt.text().strip(),
            "--device", self.cb_sim_device.currentText(),
            "--batch", str(int(self.sp_sim_batch.value())),
            "--input_size", str(int(self.sp_sim_in.value())),
            "--strict_align", "1" if self.chk_strict.isChecked() else "0",
        ]
        return cmd

    def build_commands(self, plan: RunPlan) -> List[List[str]]:
        cmds: List[List[str]] = []
        if plan.enable_clip:
            cmds.append(self.build_clip_cmd(plan))
        if plan.enable_simclr:
            cmds.append(self.build_simclr_cmd())
        return cmds

    # ---------- pre-run alignment check ----------
    def precheck_alignment(self, plan: RunPlan) -> bool:
        """
        True means proceed.
        Only does messaging (no blocking). The goal is:
        - If SimCLR enabled & not aligned -> prompt "need catch-up"
        - If aligned -> prompt "already aligned (can skip SimCLR)"
        """
        st = getattr(self, "_latest_status", None)
        if not plan.enable_simclr:
            return True

        # If we don't have status yet, refresh once and proceed (script itself will validate)
        if not isinstance(st, dict):
            QtWidgets.QMessageBox.information(self, "提示", "对齐状态尚未读取完成，已继续执行（建议先点“刷新状态”确认）。")
            return True

        aligned = st.get("aligned", None)
        meta_len = st.get("images_meta_len", None)
        vec_len = st.get("simclr_vec2img_len", None)

        if aligned is False:
            QtWidgets.QMessageBox.warning(
                self,
                "需要 SimCLR catch-up",
                f"检测到未对齐：\n"
                f"images_meta_len = {meta_len}\n"
                f"simclr_vec2img_len = {vec_len}\n\n"
                f"将执行 SimCLR MODE-A 追尾补齐。"
            )
            return True

        if aligned is True:
            QtWidgets.QMessageBox.information(
                self,
                "已对齐",
                f"SimCLR 已对齐：\n"
                f"images_meta_len = {meta_len}\n"
                f"simclr_vec2img_len = {vec_len}\n\n"
                f"如果你想节省时间，可以取消勾选 SimCLR。"
            )
            return True

        QtWidgets.QMessageBox.information(
            self,
            "无法判断对齐",
            "当前对齐信息不完整（读取失败或文件缺失），将继续执行。"
        )
        return True

    # ---------- run ----------
    def on_run(self):
        plan = self.collect_plan()
        if not self.validate_basic(plan):
            return

        # update log file
        self.log_file_path = self.ed_log_file.text().strip() or None
        if self.log_file_path:
            try:
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                self.log_file_path = None

        # precheck alignment提示
        self.precheck_alignment(plan)

        self.log("=== RUN PLAN ===")
        self.log(f"roots = {plan.new_roots}")
        self.log(f"clip={plan.enable_clip} | simclr={plan.enable_simclr} | dry_run={plan.dry_run}")
        self.log("NOTE: roots/out_dir 由 GUI 通过 argparse 传参；脚本内 CONFIG 仅作为默认值。")

        self.queue = self.build_commands(plan)

        self.log("=== COMMANDS ===")
        for i, c in enumerate(self.queue, 1):
            self.log(f"[{i}] " + " ".join(c))

        if plan.dry_run:
            self.log("=== DRY RUN DONE ===")
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.run_next()

    def run_next(self):
        if not self.queue:
            self.log("=== ALL DONE ===")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.worker = None
            # run finished -> refresh status
            self.refresh_status()
            return

        cmd = self.queue.pop(0)
        self.worker = ProcWorker(cmd)
        self.worker.sig_log.connect(self.log)
        self.worker.sig_done.connect(self.on_one_done)
        self.worker.start()

    def on_one_done(self, code: int):
        self.log(f"[EXIT] code={code}")
        if code != 0:
            self.log("=== STOP: command failed ===")
            self.queue = []
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            # refresh status anyway
            self.refresh_status()
            return
        self.run_next()

    def on_stop(self):
        self.log("[STOP] try kill process tree ...")
        if self.worker:
            self.worker.kill_tree_windows()
        self.queue = []
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log("[STOP] done.")
        self.refresh_status()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()