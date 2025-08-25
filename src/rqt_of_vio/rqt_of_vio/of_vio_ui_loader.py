import os
from ament_index_python import get_resource
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget


class OfVIOWidget(QWidget):
    def __init__(self):
        super(OfVIOWidget, self).__init__()
        _, package_path = get_resource('packages', 'rqt_of_vio')
        ui_file = os.path.join(package_path, 'share', 'rqt_of_vio', 'resource', 'of_vio.ui')
        loadUi(ui_file, self)
        