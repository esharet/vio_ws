#!/usr/bin/env python3

"""
Connect backend and view
listen for event from backend and update the view
"""
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
from collections import deque

class OfVIORqtPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self._widget = OfVIOWidget()
        self._backend = BackendNode()
        self._backend.start()

        self.layout = self._widget.layout
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        velocity_y_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.layout.addWidget(dynamic_canvas)
        self.layout.addWidget(velocity_y_canvas)

        self._velocity_x = dynamic_canvas.figure.subplots()
        self._velocity_y = velocity_y_canvas.figure.subplots()

        
        # Set up a Line2D.
        buffer_size = 200
        self.xdata = np.arange(buffer_size)
        self.vx_estimate = deque([0]*buffer_size, maxlen=buffer_size)
        self.vx_estimate_filtered = deque([0]*buffer_size, maxlen=buffer_size)
        self.vx_truth = deque([0]*buffer_size, maxlen=buffer_size)

        self.vy_estimate = deque([0]*buffer_size, maxlen=buffer_size)
        self.vy_estimate_filtered = deque([0]*buffer_size, maxlen=buffer_size)
        self.vy_truth = deque([0]*buffer_size, maxlen=buffer_size)

        self._vx_estimate_line, = self._velocity_x.plot(self.xdata, self.vx_estimate, label="estimate")
        self._vx_estimate_filtered_line, = self._velocity_x.plot(self.xdata, self.vx_estimate_filtered, label="estimate_filtered")
        self._vx_truth_line, = self._velocity_x.plot(self.xdata, self.vx_truth, label="truth")
        

        self._vy_estimate_line, = self._velocity_y.plot(self.xdata, self.vy_estimate, label="estimate")
        self._vy_estimate_filtered_line, = self._velocity_y.plot(self.xdata, self.vy_estimate_filtered, label="estimate_filtered")
        self._vy_truth_line, = self._velocity_y.plot(self.xdata, self.vy_truth, label="truth")

        self._velocity_x.set_title("X Vel")
        self._velocity_y.set_title("Y Vel")
        self._velocity_x.legend()
        self._velocity_y.legend()
        # The below two timers must be attributes of self, so that the garbage
        # collector won't clean them after we finish with __init__...

        # The data retrieval may be fast as possible (Using QRunnable could be
        # even faster).
        # self.data_timer = dynamic_canvas.new_timer(1)
        # self.data_timer.add_callback(self._backend.sim_estimate_velocity)
        # self.data_timer.start()
        # Drawing at 50Hz should be fast enough for the GUI to feel smooth, and
        # not too fast for the GUI to be overloaded with events that need to be
        # processed while the GUI element is changed.
        self.drawing_timer = dynamic_canvas.new_timer(20)
        self.drawing_timer.add_callback(self._update_canvas)
        self.drawing_timer.start()
        self.t = 0
        self._backend.on_estimate_velocity += self.estimate_velocity_handler
        self._backend.on_estimate_velocity_filtered += self.__estimate_velocity_filtered_handler
        self._backend.on_truth_velocity += self.truth_velocity_handler
        context.add_widget(self._widget)
        
    def _update_canvas(self):
        # update matplotlib canvas
        self._vx_estimate_line.set_ydata(self.vx_estimate)
        self._vx_estimate_filtered_line.set_ydata(self.vx_estimate_filtered)
        self._vx_truth_line.set_ydata(self.vx_truth)

        self._vy_estimate_line.set_ydata(self.vy_estimate)
        self._vy_estimate_filtered_line.set_ydata(self.vy_estimate_filtered)
        self._vy_truth_line.set_ydata(self.vy_truth)

        self._vx_estimate_line.axes.relim()            # recompute limits
        self._vx_estimate_line.axes.autoscale_view()   # adjust view to fit new data
        self._vx_estimate_line.figure.canvas.draw_idle()

        self._vy_estimate_line.axes.relim()            # recompute limits
        self._vy_estimate_line.axes.autoscale_view()   # adjust view to fit new data
        self._vy_estimate_line.figure.canvas.draw_idle()

        
    def truth_velocity_handler(self, vx, vy):
        self.vx_truth.append(vx)
        self.vy_truth.append(vy)

    def estimate_velocity_handler(self, vx, vy):
        self.vx_estimate.append(vx)
        self.vy_estimate.append(vy)
    
    def __estimate_velocity_filtered_handler(self, vx, vy):
        self.vx_estimate_filtered.append(vx)
        self.vy_estimate_filtered.append(vy)
    
