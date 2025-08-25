#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from rqt_gui_py.plugin import Plugin
from PyQt5.QtCore import QStringListModel, Qt, QTimer, QSize
# from python_qt_binding.QtWidgets import QApplication, QWidget, QVBoxLayout, QListView, QPushButton
from PyQt5.QtWidgets import (
    QPushButton, 
    QListView, 
    QAbstractItemView, 
    QMessageBox, 
    QLineEdit, 
    QFileDialog,
    QRadioButton,
    QComboBox,
    QLabel,
    QDialog,
    QVBoxLayout,
    QHBoxLayout
    )
from PyQt5.QtGui import QPixmap, QIcon
from pathlib import Path
from functools import partial
from .of_vio_ui_loader import OfVIOWidget
from .of_vio_backend import BackendNode
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import time


class OfVIORqtPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self._widget = OfVIOWidget()
        self._backend = BackendNode()

        self.layout = self._widget.layout
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.layout.addWidget(dynamic_canvas)

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._dynamic_ax.set_title("Dynamic plot")
        # Set up a Line2D.
        self.xdata = np.linspace(0, 10, 101)
        self._update_ydata()
        self._line, = self._dynamic_ax.plot(self.xdata, self.ydata)
        # The below two timers must be attributes of self, so that the garbage
        # collector won't clean them after we finish with __init__...

        # The data retrieval may be fast as possible (Using QRunnable could be
        # even faster).
        self.data_timer = dynamic_canvas.new_timer(1)
        self.data_timer.add_callback(self._update_ydata)
        self.data_timer.start()
        # Drawing at 50Hz should be fast enough for the GUI to feel smooth, and
        # not too fast for the GUI to be overloaded with events that need to be
        # processed while the GUI element is changed.
        self.drawing_timer = dynamic_canvas.new_timer(20)
        self.drawing_timer.add_callback(self._update_canvas)
        self.drawing_timer.start()

        context.add_widget(self._widget)
        
    def _update_canvas(self):
        # update matplotlib canvas
        self._line.set_data(self.xdata, self.ydata)
        self._line.figure.canvas.draw_idle()

    def _update_ydata(self):
        # update the data only
        self.ydata = np.sin(self.xdata + time.time())