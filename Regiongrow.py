import sys
import os
import numpy as np
import cv2
from yacs.config import CfgNode
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


def seedfill(f, seeds, conn=8):
    """

    :param f: binary image
    :param seeds: list or numpy array with shape (2) or (N, 2)
    :param conn: 4 or 8
    :return: g: binary image
    """
    # 参数检查
    seeds = np.asarray(seeds, dtype=np.int32)
    assert len(f.shape) == 2, '输入图像f必须为二值图'
    assert (len(seeds.shape) in [1, 2]) and seeds.shape[-1] == 2, '输入种子点形状必须为(2)或者(N, 2)'
    assert conn in [4, 8], '邻域类型必须为4或8'
    height, width = f.shape[0], f.shape[1]

    # 邻域
    if conn == 8:
        dxy = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0],
                        [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=int)
    else:
        dxy = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]], dtype=int)

    # 算法
    g = np.zeros((height, width), dtype=np.uint8)
    if len(seeds.shape) == 1:
        seeds = seeds[None]

    stack = []
    for seed in seeds:
        if g[seed[1], seed[0]]:
            continue
        stack.append(seed)
        g[seed[1], seed[0]] = 255  # 标记为已处理
        while stack:
            pt = stack.pop(-1)  # 弹出栈顶元素
            # 遍历邻域
            npts = pt[None] + dxy
            mask = np.logical_and(np.logical_and(0 <= npts[:, 0], npts[:, 0] < width),
                                  np.logical_and(0 <= npts[:, 1], npts[:, 1] < height))
            npts = npts[mask]
            for npt in npts:
                if f[npt[1], npt[0]] and not g[npt[1], npt[0]]:
                    stack.append(npt)  # 种子压栈
                    g[npt[1], npt[0]] = 255  # 标记为已处理

    return g


def regiongrow(f, seed, seedvalue, thresh, conn=8):
    """

    :param f: BGR image or gray image
    :param seed: list or numpy array with shape (2)
    :param seedvalue: scalar or [b, g, r] vector
    :param thresh: scalar or [b, g, r] vector
    :param conn: 4 or 8
    :return: g: binary image
    """
    # 参数检查
    if len(f.shape) == 2:
        f = cv2.merge([f, f, f])
    if seed is not None:
        seed = np.asarray(seed)
        assert len(seed.shape) == 1 and seed.shape[-1] == 2, '输入种子点形状必须为(2)'
    else:
        assert seedvalue is not None, 'seed和seedvalue不能同时为None'
    assert conn in [4, 8], '邻域类型必须为4或8'

    # 参数seed
    if seed is not None:
        seedvalue = f[seed[1], seed[0]]
    else:
        seedvalue = np.asarray(seedvalue)
        seed = np.where(np.all(f == seedvalue, axis=2))
        seed = np.hstack((seed[1][..., None], seed[0][..., None]))
    seed = np.int32(np.round(seed))

    # 阈值处理
    thresh = np.asarray(thresh)
    bw = np.all(np.abs(f.astype(np.int32) - seedvalue) <= thresh, axis=2)

    # 种子填充
    g = seedfill(bw, seed, conn)

    return g


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()

        # Variables
        self.file = None
        self.image = None
        self.last_label = None
        self.label = None
        self.seed = {'Position': None, 'Value': None, 'Threshold': cfg.init_threshold}
        self.status = {'Position': None, 'Value': None, 'Zoom': cfg.init_scale}
        self.mouse = {'LeftButtonPress': False, 'Position': None, 'Value': None}

        # Parameters
        self.conn = cfg.conn
        self.scale = cfg.init_scale
        self.scale_fit_window = cfg.scale_fit_window
        self.scale_list = cfg.scale_list
        self.step_list = cfg.step_list

        # UI
        self.menuBar = QMenuBar()
        self.menu_File = self.menuBar.addMenu('File')
        self.menu_Edit = self.menuBar.addMenu('Edit')
        self.menu_Help = self.menuBar.addMenu('Help')
        self.menu_Open = self.menu_File.addAction(QIcon('icon/open.png'), 'Open')
        self.menu_File.addSeparator()
        self.menu_Save = self.menu_File.addAction(QIcon('icon/save.png'), 'Save')
        self.menu_Run = self.menu_Edit.addAction(QIcon('icon/run.png'), 'Run')
        self.menu_Edit.addSeparator()
        self.menu_Undo = self.menu_Edit.addAction(QIcon('icon/undo.png'), 'Undo')
        self.menu_Edit.addSeparator()
        self.menu_Delete = self.menu_Edit.addAction(QIcon('icon/delete.png'), 'Delete')
        self.menu_Tutorial = self.menu_Help.addAction(QIcon('icon/tutorial.png'), 'Tutorial')

        self.toolBar = QToolBar()
        self.tool_Open = self.toolBar.addAction(QIcon('icon/open.png'), 'Open')
        self.tool_Save = self.toolBar.addAction(QIcon('icon/save.png'), 'Save')
        self.toolBar.addSeparator()
        self.tool_Run = self.toolBar.addAction(QIcon('icon/run.png'), 'Run')
        self.tool_Undo = self.toolBar.addAction(QIcon('icon/undo.png'), 'Undo')
        self.tool_Delete = self.toolBar.addAction(QIcon('icon/delete.png'), 'Delete')
        self.toolBar.addSeparator()
        self.tool_ZoomIn = self.toolBar.addAction(QIcon('icon/zoom-in.png'), 'Zoom In')
        self.tool_ZoomOut = self.toolBar.addAction(QIcon('icon/zoom-out.png'), 'Zoom Out')

        self.statusbar = QStatusBar()

        self.groupbox_Seed = QGroupBox('Seed')
        self.button_Value = QToolButton()

        self.text_Position = QLabel('Position')
        self.text_Value = QLabel('Value')
        self.text_Threshold = QLabel('Threshold')
        self.edit_Position = QLineEdit('[]', alignment=Qt.AlignCenter)
        self.edit_Position.setStyleSheet(f'background-color:rgb(255, 255, 255)')
        self.edit_Value = QLineEdit('[]', alignment=Qt.AlignCenter)
        self.edit_Threshold = QLineEdit('[]', alignment=Qt.AlignCenter)
        self.edit_Value.setValidator(QRegExpValidator(QRegExp('((\d+)|((\d)+,\s(\d)+,\s(\d)+))|(\[((\d+)|'
                                                              '((\d)+,\s(\d)+,\s(\d)+))\])')))
        self.edit_Threshold.setValidator(QRegExpValidator(QRegExp('((\d+)|((\d)+,\s(\d)+,\s(\d)+))|(\[((\d+)|'
                                                              '((\d)+,\s(\d)+,\s(\d)+))\])')))

        self.scrollarea_Image = QScrollArea()
        self.scrollarea_Label = QScrollArea()

        self.figure_Image = QLabel()
        self.figure_Label = QLabel()

        self.ValueLayout = QHBoxLayout()
        self.SeedLayout = QVBoxLayout()
        self.EditLayout = QVBoxLayout()
        self.imageLayout = QVBoxLayout()
        self.labelLayout = QVBoxLayout()

        self.layout = QHBoxLayout()
        self.centralWidget = QWidget()

        self.InitUI(cfg)

    def InitUI(self, cfg):
        # Set UI
        self.ValueLayout.addWidget(self.text_Value)
        self.ValueLayout.addStretch(1)
        self.ValueLayout.addWidget(self.button_Value)
        self.ValueLayout.addStretch(1)

        self.SeedLayout.addWidget(self.text_Position)
        self.SeedLayout.addWidget(self.edit_Position)
        self.SeedLayout.addLayout(self.ValueLayout)
        self.SeedLayout.addWidget(self.edit_Value)
        self.SeedLayout.addWidget(self.text_Threshold)
        self.SeedLayout.addWidget(self.edit_Threshold)
        self.groupbox_Seed.setLayout(self.SeedLayout)

        self.EditLayout.addStretch(1)
        self.EditLayout.addWidget(self.groupbox_Seed)
        self.EditLayout.addStretch(1)

        self.figure_Image.setScaledContents(True)
        self.figure_Label.setScaledContents(True)

        self.scrollarea_Image.setWidget(self.figure_Image)
        self.scrollarea_Image.setAlignment(Qt.AlignCenter)
        self.scrollarea_Image.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollarea_Image.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.imageLayout.addWidget(self.scrollarea_Image)

        self.scrollarea_Label.setWidget(self.figure_Label)
        self.scrollarea_Label.setAlignment(Qt.AlignCenter)
        self.scrollarea_Label.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollarea_Label.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollarea_Label.horizontalScrollBar().setEnabled(False)
        self.scrollarea_Label.verticalScrollBar().setEnabled(False)
        self.labelLayout.addWidget(self.scrollarea_Label)

        self.layout.addLayout(self.EditLayout)
        self.layout.addLayout(self.imageLayout)
        self.layout.addLayout(self.labelLayout)

        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)

        self.setMenuBar(self.menuBar)
        self.addToolBar(self.toolBar)
        self.setStatusBar(self.statusbar)

        self.setWindowTitle('Regiongrow')
        self.resize(cfg.window_size[0], cfg.window_size[1])
        self.groupbox_Seed.setFixedSize(self.groupbox_Seed.sizeHint())

        # Register callback
        self.menu_Open.triggered.connect(self.Open_Callback)
        self.menu_Save.triggered.connect(self.Save_Callback)
        self.menu_Run.triggered.connect(self.Run_Callback)
        self.menu_Undo.triggered.connect(self.Undo_Callback)
        self.menu_Delete.triggered.connect(self.Delete_Callback)
        self.menu_Tutorial.triggered.connect(self.Tutorial_Callback)

        self.tool_Open.triggered.connect(self.Open_Callback)
        self.tool_Save.triggered.connect(self.Save_Callback)
        self.tool_Run.triggered.connect(self.Run_Callback)
        self.tool_Undo.triggered.connect(self.Undo_Callback)
        self.tool_Delete.triggered.connect(self.Delete_Callback)
        self.tool_ZoomIn.triggered.connect(self.ZoomIn_Callback)
        self.tool_ZoomOut.triggered.connect(self.ZoomOut_Callback)

        self.button_Value.clicked.connect(self.Value_Callback)
        self.button_Value.setAutoRaise(True)

        self.edit_Value.editingFinished.connect(self.EditValue_Callback)
        self.edit_Threshold.editingFinished.connect(self.EditThreshold_Callback)

        self.scrollarea_Image.horizontalScrollBar().valueChanged.connect(self.horizontalMoveEvent_Callback)
        self.scrollarea_Image.verticalScrollBar().valueChanged.connect(self.verticalMoveEvent_Callback)

        self.figure_Image.mousePressEvent = self.mousePress_Callback
        self.figure_Image.mouseReleaseEvent = self.mouseRelease_Callback
        self.figure_Image.mouseMoveEvent = self.mouseMove_Callback
        self.figure_Image.wheelEvent = self.wheelEvent_Callback
        self.figure_Image.setMouseTracking(True)

        # Add shortcut
        self.menu_Open.setShortcut(cfg.menu_Open_shortcut)
        self.menu_Save.setShortcut(cfg.menu_Save_shortcut)
        self.menu_Run.setShortcut(cfg.menu_Run_shortcut)
        self.menu_Undo.setShortcut(cfg.menu_Undo_shortcut)
        self.menu_Delete.setShortcut(cfg.menu_Delete_shortcut)
        self.menu_Tutorial.setShortcut(cfg.menu_Tutorial_shortcut)

        # Update Widget
        self.reset()
        self.menu_Open.setEnabled(True)
        self.tool_Open.setEnabled(True)

    def event(self, QEvent):
        if QEvent.type() == QEvent.StatusTip:
            if QEvent.tip() == '':
                QEvent = QStatusTipEvent(self.update_statusbar())
        return super().event(QEvent)

    def reset(self):
        self.menu_Open.setEnabled(False)
        self.menu_Save.setEnabled(False)
        self.menu_Run.setEnabled(False)
        self.menu_Undo.setEnabled(False)
        self.menu_Delete.setEnabled(False)

        self.tool_Open.setEnabled(False)
        self.tool_Save.setEnabled(False)
        self.tool_Run.setEnabled(False)
        self.tool_Undo.setEnabled(False)
        self.tool_Delete.setEnabled(False)
        self.tool_ZoomIn.setEnabled(False)
        self.tool_ZoomOut.setEnabled(False)

        self.button_Value.setEnabled(False)

        self.edit_Position.setEnabled(False)
        self.edit_Value.setEnabled(False)
        self.edit_Threshold.setEnabled(False)

        self.figure_Image.setEnabled(False)

    def update_seed_info(self):
        position = self.seed['Position']
        if position is None:
            self.edit_Position.setText('')
        else:
            self.edit_Position.setText(f'[{position[0]}, {position[1]}]')

        value = self.seed['Value']
        if value is None:
            self.edit_Value.setText('')
            self.button_Value.setStyleSheet(f'background-color:rgb(128, 128, 128)')
        else:
            self.edit_Value.setText(f'[{value[0]}, {value[1]}, {value[2]}]')
            self.button_Value.setStyleSheet(f'background-color:rgb({value[2]}, {value[1]}, {value[0]})')

        threshold = self.seed['Threshold']
        self.edit_Threshold.setText(f'[{threshold[0]}, {threshold[1]}, {threshold[2]}]')

    def update_statusbar(self):
        position_smg, value_msg, zoom_msg = '', '', ''
        position = self.status['Position']
        if position is not None:
            position_smg = f'Position: [{position[0]}, {position[1]}], '

        value = self.status['Value']
        if value is not None:
            value_msg = f'Value: [{value[0]}, {value[1]}, {value[2]}], '

        zoom = self.status['Zoom']
        zoom_msg = f'Zoom: {int(round(zoom * 100))}%'

        message = position_smg + value_msg + zoom_msg
        self.statusbar.showMessage(message)
        return message

    def update_figure(self):
        # Plot image
        image = self.image.copy()
        new_size = (int(round(image.shape[1] * self.scale)), int(round(image.shape[0] * self.scale)))
        image = cv2.resize(image, new_size, cv2.INTER_CUBIC)
        image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap(image).scaled(new_size[0], new_size[1])
        self.figure_Image.setPixmap(pixmap)
        self.figure_Image.resize(new_size[0], new_size[1])

        # Plot label
        label = self.label.copy()
        label = cv2.resize(label, new_size, cv2.INTER_CUBIC)
        label = QImage(label.data, label.shape[1], label.shape[0], label.shape[1], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap(label).scaled(new_size[0], new_size[1])
        self.figure_Label.setPixmap(pixmap)
        self.figure_Label.resize(new_size[0], new_size[1])

    def update_widget(self):
        has_result = bool(np.any(self.label))
        has_last_result = self.last_label is not None
        runable = self.seed['Position'] is not None or self.seed['Value'] is not None

        self.menu_Open.setEnabled(True)
        self.menu_Save.setEnabled(has_result)
        self.menu_Run.setEnabled(runable)
        self.menu_Undo.setEnabled(has_last_result)
        self.menu_Delete.setEnabled(has_result)

        self.tool_Open.setEnabled(True)
        self.tool_Save.setEnabled(has_result)
        self.tool_Run.setEnabled(runable)
        self.tool_Undo.setEnabled(has_last_result)
        self.tool_Delete.setEnabled(has_result)
        self.tool_ZoomIn.setEnabled(self.scale < self.scale_list[-1])
        self.tool_ZoomOut.setEnabled(self.scale > self.scale_list[0])

        self.button_Value.setEnabled(True)

        self.edit_Value.setEnabled(True)
        self.edit_Threshold.setEnabled(True)

        self.figure_Image.setEnabled(True)

    def Open_Callback(self):
        file = QFileDialog.getOpenFileName(caption='Open image file', filter='Image Files(*.jpg *.png);;All Files(*)')[0]
        if not os.path.isfile(file):
            return

        image = cv2.imread(file)
        if image is None:
            QMessageBox.warning(self, 'Warning', 'Please select a valid image file!', QMessageBox.Close)
            return

        if len(image.shape) == 2:
            image = cv2.merge([image, image, image])
        self.file = file
        self.image = image
        self.last_label = None
        self.label = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        if self.scale_fit_window:
            width = self.scrollarea_Image.width() - 50
            height = self.scrollarea_Image.height() - 50
            image_width, image_height = self.image.shape[1], self.image.shape[0]
            scale_x = width / image_width
            scale_y = height / image_height
            scale = min(scale_x, scale_y)
            self.scale = float(np.clip(scale, self.scale_list[0], self.scale_list[-1]))
            self.status['Zoom'] = self.scale

        # Update UI
        self.update_seed_info()
        self.update_statusbar()
        self.update_figure()

        # Update Widget
        self.update_widget()

    def Save_Callback(self):
        directory = os.path.dirname(self.file)
        file = QFileDialog.getSaveFileName(caption='Save image file', directory=directory,
                                           filter='Image Files(*.jpg *.png);;All Files(*)')[0]
        if file != '':
            cv2.imwrite(file, self.label)

    def Run_Callback(self):
        # Update Widget
        self.reset()

        # Run
        label = regiongrow(self.image, self.seed['Position'], self.seed['Value'], self.seed['Threshold'], self.conn)
        self.last_label = self.label.copy()
        self.label[label > 0] = 255

        # Update UI
        self.update_figure()

        # Update Widget
        self.update_widget()

    def Undo_Callback(self):
        self.label = self.last_label.copy()
        self.last_label = None

        # Update UI
        self.update_figure()

        # Update Widget
        self.update_widget()

    def Delete_Callback(self):
        self.label = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)

        # Update UI
        self.update_figure()

        # Update Widget
        self.update_widget()

    def ZoomIn_Callback(self):
        mask = [self.scale < scale for scale in self.scale_list]
        index = mask.index(True)
        self.scale = self.scale_list[index]
        self.status['Zoom'] = self.scale

        # Update UI
        self.update_statusbar()
        self.update_figure()

        # Update Widget
        self.update_widget()

    def ZoomOut_Callback(self):
        mask = [self.scale <= scale for scale in self.scale_list]
        index = mask.index(True) - 1
        self.scale = self.scale_list[index]
        self.status['Zoom'] = self.scale

        # Update UI
        self.update_statusbar()
        self.update_figure()

        # Update Widget
        self.update_widget()

    def Tutorial_Callback(self):
        QMessageBox.information(self,
                        'Tutorial',
                        'Version: 1.0\n'
                        'Author: lh9171338\n'
                        'Date: 2020-12-31\n'
                        'Shortcut (default):\n'
                        '\tCtrl + O: Open an image file\n'
                        '\tCtrl + S: Save the result\n'
                        '\tF5: Run the algorithm\n'
                        '\tCtrl + Z: Undo the last operation\n'
                        '\tCtrl + D: Delete all the previous operations\n'
                        '\tCtrl + T: View the tutorial\n'
                        'Mouse (Operate in the image area): \n'
                        '\tLeft button drag: Move the image\n' 
                        '\tMouse wheel: Zoom in or zoom out\n'
                        '\tRight button click: Select a seed',
                        QMessageBox.Close)

    def Value_Callback(self):
        value = self.seed['Value']
        if value is None:
            init_color = QColor(128, 128, 128)
        else:
            init_color = QColor(value[2], value[1], value[0])
        color = QColorDialog.getColor(init_color).getRgb()
        value = color[:3][::-1]
        self.seed['Value'] = value
        self.seed['Position'] = None

        # Update UI
        self.update_seed_info()

        # Update Widget
        self.update_widget()

    def EditValue_Callback(self):
        text = self.edit_Value.text().strip('[').strip(']').split(',')
        value = [int(np.clip(int(t), 0, 255)) for t in text]
        if len(value) == 1:
            value = value * 3
        self.seed['Value'] = value
        self.seed['Position'] = None

        # Update UI
        self.update_seed_info()

        # Update Widget
        self.update_widget()

    def EditThreshold_Callback(self):
        text = self.edit_Threshold.text().strip('[').strip(']').split(',')
        threshold = [int(np.clip(int(t), 0, 255)) for t in text]
        if len(threshold) == 1:
            threshold = threshold * 3
        self.seed['Threshold'] = threshold

        # Update UI
        self.update_seed_info()

    def horizontalMoveEvent_Callback(self):
        value = self.scrollarea_Image.horizontalScrollBar().value()
        self.scrollarea_Label.horizontalScrollBar().setValue(value)

    def verticalMoveEvent_Callback(self):
        value = self.scrollarea_Image.verticalScrollBar().value()
        self.scrollarea_Label.verticalScrollBar().setValue(value)

    def mousePress_Callback(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse['LeftButtonPress'] = True
            self.mouse['Position'] = [event.globalX(), event.globalY()]
            self.mouse['Value'] = [self.scrollarea_Image.horizontalScrollBar().value(),
                                   self.scrollarea_Image.verticalScrollBar().value()]
            self.setCursor(Qt.OpenHandCursor)

    def mouseRelease_Callback(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse['LeftButtonPress'] = False
            self.mouse['Position'] = None
            self.mouse['Value'] = None
            self.setCursor(Qt.ArrowCursor)

        elif event.button() == Qt.RightButton:
            width, height = self.image.shape[1], self.image.shape[0]
            x = int(round(event.x() / self.scale))
            y = int(round(event.y() / self.scale))
            if x < 0 or x >= width or y < 0 or y >= height:
                return

            value = self.image[y, x]
            self.seed['Position'] = [x, y]
            self.seed['Value'] = [value[0], value[1], value[2]]

            # Update UI
            self.update_seed_info()

            # Update Widget
            self.update_widget()

    def mouseMove_Callback(self, event):
        if self.mouse['LeftButtonPress']:
            x, y = event.globalX(), event.globalY()
            x = self.mouse['Position'][0] - x + self.mouse['Value'][0]
            y = self.mouse['Position'][1] - y + self.mouse['Value'][1]

            x_min = self.scrollarea_Image.horizontalScrollBar().minimum()
            x_max = self.scrollarea_Image.horizontalScrollBar().maximum()
            y_min = self.scrollarea_Image.verticalScrollBar().minimum()
            y_max = self.scrollarea_Image.verticalScrollBar().maximum()
            x = int(np.clip(x, x_min, x_max))
            y = int(np.clip(y, y_min, y_max))

            # Update UI
            self.scrollarea_Image.horizontalScrollBar().setValue(x)
            self.scrollarea_Image.verticalScrollBar().setValue(y)
        else:
            width, height = self.image.shape[1], self.image.shape[0]
            x = int(round(event.x() / self.scale))
            y = int(round(event.y() / self.scale))
            if x < 0 or x >= width or y < 0 or y >= height:
                return

            value = self.image[y, x]
            self.status['Position'] = [x, y]
            self.status['Value'] = [value[0], value[1], value[2]]

            # Update UI
            self.update_statusbar()

    def wheelEvent_Callback(self, event):
        mask = [self.scale <= scale for scale in self.scale_list]
        index = mask.index(True) - 1
        step = self.step_list[index]
        if event.angleDelta().y() > 0:
            self.scale += step
        else:
            self.scale -= step
        self.scale = float(np.clip(self.scale, self.scale_list[0], self.scale_list[-1]))
        self.status['Zoom'] = self.scale

        # Update UI
        self.update_statusbar()
        self.update_figure()

        # Update Widget
        self.update_widget()


if __name__ == "__main__":
    cfg = CfgNode.load_cfg(open('default.yaml'))
    cfg.freeze()
    print(cfg)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    window.Open_Callback()
    sys.exit(app.exec_())
