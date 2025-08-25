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

class OfVIORqtPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self._widget = OfVIOWidget()
        self._backend = BackendNode()