import sys
import os

# 先导入 torch 以检查环境
try:
    import torch
except ImportError:
    print("错误: 未找到 torch 模块，请确保已安装 PyTorch")
    sys.exit(1)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QTableWidget, QTableWidgetItem,
                             QMessageBox, QGroupBox, QComboBox, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# 引入 mmpretrain 相关模块
try:
    from mmpretrain.apis import ImageClassificationInferencer
except ImportError:
    print("错误: 未找到 mmpretrain 模块，请确保已安装 openmim 和 mmpretrain")
    sys.exit(1)


def check_cuda_available():
    """安全地检查 CUDA 是否可用"""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


class InferenceThread(QThread):
    """
    在后台线程运行推理，避免界面卡死
    """
    finished_signal = pyqtSignal(dict, bool)  # (result_dict, success_flag)
    error_signal = pyqtSignal(str)

    def __init__(self, img_path, model_name, checkpoint_path, device):
        super().__init__()
        self.img_path = img_path
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self):
        try:
            # 构建 Inferencer
            # 注意：如果用户只输入了模型名，pretrained=True 会自动下载权重
            # 如果指定了 checkpoint 路径，则使用该路径
            pretrained = True if not self.checkpoint_path else self.checkpoint_path

            # 显式指定 device，防止 mmpretrain 自动判断出错
            inferencer = ImageClassificationInferencer(
                self.model_name,
                pretrained=pretrained,
                device=self.device
            )

            # 执行推理
            # return_predictions=True 确保返回字典格式
            results = inferencer(self.img_path)
            print("-" * 30)
            print(f"原始返回结果类型: {type(results)}")
            if len(results) > 0:
                print(f"第一个结果的内容: {results[0]}")
                print(f"第一个结果的键: {results[0].keys() if isinstance(results[0], dict) else 'Not a dict'}")
                if 'pred_scores' in results[0]:
                    print(f"pred_scores 类型: {type(results[0]['pred_scores'])}")
                    print(f"pred_scores 内容: {results[0]['pred_scores']}")
            print("-" * 30)
            if results and len(results) > 0:
                # 提取第一个结果 (单张图片)
                result_data = results[0]
                self.finished_signal.emit(result_data, True)
            else:
                self.error_signal.emit("推理结果为空")

        except Exception as e:
            # 捕获异常并发送信号
            self.error_signal.emit(str(e))


class MMPretrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('OpenMMLab 图像分类助手 (MMPretrain)')
        self.setGeometry(100, 100, 900, 700)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # --- 顶部控制区 ---
        control_group = QGroupBox("模型与设置")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)

        # 模型名称行
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型名称/配置路径:"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("例如: resnet50_8xb32_in1k 或 config.py 路径")
        self.model_input.setText("resnet50_8xb32_in1k")  # 默认值
        model_layout.addWidget(self.model_input)
        control_layout.addLayout(model_layout)

        # Checkpoint 和设备行
        param_layout = QHBoxLayout()

        param_layout.addWidget(QLabel("Checkpoint (可选):"))
        self.ckpt_input = QLineEdit()
        self.ckpt_input.setPlaceholderText("留空则自动下载预训练权重")
        btn_browse_ckpt = QPushButton("浏览...")
        btn_browse_ckpt.clicked.connect(self.browse_checkpoint)
        param_layout.addWidget(self.ckpt_input)
        param_layout.addWidget(btn_browse_ckpt)

        param_layout.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "mps"])  # mps for Mac

        # 【修复点】使用正确的函数检查 CUDA
        has_cuda = check_cuda_available()
        if has_cuda:
            self.device_combo.setCurrentText("cuda")
        else:
            self.device_combo.setCurrentText("cpu")

        param_layout.addWidget(self.device_combo)

        control_layout.addLayout(param_layout)

        # 选择图片按钮
        btn_layout = QHBoxLayout()
        self.btn_select_img = QPushButton("📂 选择图片")
        self.btn_select_img.clicked.connect(self.select_image)
        self.btn_run = QPushButton("🚀 开始分类")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_run.setEnabled(False)  # 初始禁用，直到选择图片

        btn_layout.addWidget(self.btn_select_img)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addStretch()
        control_layout.addLayout(btn_layout)

        main_layout.addWidget(control_group)

        # --- 中间显示区 (图片 + 结果) ---
        content_layout = QHBoxLayout()

        # 左侧：图片预览
        img_group = QGroupBox("图片预览")
        img_layout = QVBoxLayout()
        img_group.setLayout(img_layout)
        self.img_label = QLabel("暂无图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("QLabel { background-color: #f0f0f0; color: #888; min-height: 300px; }")
        self.img_label.setMinimumWidth(300)
        img_layout.addWidget(self.img_label)
        content_layout.addWidget(img_group, stretch=1)

        # 右侧：结果表格
        result_group = QGroupBox("分类结果")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["类别 (Class)", "置信度 (Score)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        result_layout.addWidget(self.result_table)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: gray;")
        result_layout.addWidget(self.status_label)

        content_layout.addWidget(result_group, stretch=2)
        main_layout.addLayout(content_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_run.setEnabled(True)
            self.status_label.setText(f"已加载: {os.path.basename(file_path)}")

    def display_image(self, path):
        pixmap = QPixmap(path)
        # 缩放以适应标签大小，保持宽高比
        scaled_pixmap = pixmap.scaled(
            self.img_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.img_label.setPixmap(scaled_pixmap)

    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Checkpoint", "", "Checkpoint Files (*.pth *.pt)"
        )
        if file_path:
            self.ckpt_input.setText(file_path)

    def run_inference(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择一张图片！")
            return

        model_name = self.model_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "警告", "请输入模型名称或配置路径！")
            return

        ckpt_path = self.ckpt_input.text().strip()
        device = self.device_combo.currentText()

        # 二次检查：如果用户手动选了 cuda 但实际没有，提示警告（可选）
        if device == 'cuda' and not check_cuda_available():
            reply = QMessageBox.question(self, '确认',
                                         '未检测到 CUDA 设备，但您选择了 cuda。\n是否强制尝试？(可能会报错)',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self.device_combo.setCurrentText("cpu")
                device = "cpu"

        # 锁定界面
        self.btn_run.setEnabled(False)
        self.btn_run.setText("推理中...")
        self.result_table.setRowCount(0)
        self.status_label.setText("正在加载模型并推理，请稍候...")

        # 启动后台线程
        self.thread = InferenceThread(self.current_image_path, model_name, ckpt_path, device)
        self.thread.finished_signal.connect(self.on_inference_finished)
        self.thread.error_signal.connect(self.on_inference_error)
        self.thread.start()

    def on_inference_finished(self, result, success):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 开始分类")

        if success:
            self.status_label.setText("推理完成")

            # 清空表格
            self.result_table.setRowCount(0)

            # --- 核心逻辑修改开始 ---
            # 情况 A: 新版本/自定义模型直接返回了 pred_class 和 pred_score (最理想情况)
            if 'pred_class' in result and 'pred_score' in result:
                top_class = result['pred_class']
                top_score = float(result['pred_score'])

                # 在表格中只显示最佳结果 (因为其他类别的名字未知，除非你有 meta 文件)
                self.result_table.setRowCount(1)

                # 设置类别
                item_cls = QTableWidgetItem(str(top_class))
                item_cls.setFlags(item_cls.flags() & ~Qt.ItemIsEditable)
                # 高亮样式
                item_cls.setBackground(Qt.lightGray)
                font = item_cls.font()
                font.setBold(True)
                item_cls.setFont(font)
                self.result_table.setItem(0, 0, item_cls)

                # 设置分数
                item_score = QTableWidgetItem(f"{top_score:.4f}")
                item_score.setFlags(item_score.flags() & ~Qt.ItemIsEditable)
                item_score.setBackground(Qt.lightGray)
                item_score.setFont(font)
                self.result_table.setItem(0, 1, item_score)

                self.status_label.setText(f"识别结果：{top_class} (置信度: {top_score:.2%})")
                return

            # 情况 B: 旧版本或标准模型返回的是字典 {class_name: score}
            predictions = result.get('pred_scores', {})
            if isinstance(predictions, dict) and len(predictions) > 0:
                sorted_classes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                self.result_table.setRowCount(len(sorted_classes))

                for i, (cls_name, score) in enumerate(sorted_classes):
                    item_cls = QTableWidgetItem(str(cls_name))
                    item_cls.setFlags(item_cls.flags() & ~Qt.ItemIsEditable)
                    self.result_table.setItem(i, 0, item_cls)

                    try:
                        score_val = float(score)
                        item_score = QTableWidgetItem(f"{score_val:.4f}")
                    except (ValueError, TypeError):
                        item_score = QTableWidgetItem(str(score))
                    item_score.setFlags(item_score.flags() & ~Qt.ItemIsEditable)
                    self.result_table.setItem(i, 1, item_score)

                    if i == 0:
                        item_cls.setBackground(Qt.lightGray)
                        item_score.setBackground(Qt.lightGray)
                        font = item_cls.font()
                        font.setBold(True)
                        item_cls.setFont(font)
                        item_score.setFont(font)
                return

            # 情况 C: 只有数组没有类名 (兜底处理)
            if isinstance(predictions, np.ndarray):
                self.status_label.setText("检测到分数数组但未找到类别名称映射。")
                # 尝试显示索引和分数
                if 'pred_label' in result:
                    idx = int(result['pred_label'])
                    score = float(predictions[idx]) if idx < len(predictions) else 0.0
                    self.result_table.setRowCount(1)
                    self.result_table.setItem(0, 0, QTableWidgetItem(f"类别索引：{idx}"))
                    self.result_table.setItem(0, 1, QTableWidgetItem(f"{score:.4f}"))
                return

            # 如果都没匹配到
            self.status_label.setText("未检测到有效的预测结果格式")
            # --- 核心逻辑修改结束 ---

        else:
            self.status_label.setText("推理失败")
    def on_inference_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 开始分类")
        self.status_label.setText("发生错误")
        QMessageBox.critical(self, "推理错误", f"发生以下错误:\n{error_msg}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 简单的样式美化
    app.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #45a049; }
        QPushButton:disabled { background-color: #cccccc; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            margin-top: 10px;
            padding-top: 10px;
            border-radius: 5px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QTableWidget { gridline-color: #ddd; }
        QLineEdit { padding: 5px; border: 1px solid #ccc; border-radius: 3px; }
    """)

    window = MMPretrainGUI()
    window.show()
    sys.exit(app.exec_())