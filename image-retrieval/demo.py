from retrieval.create_thumb_images import create_thumb_images
from retrieval.retrieval import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query
import infer_on_single_image as code_base
import os
import cv2
import time

import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

#Create thumb images
# create_thumb_images(full_folder='./static/image_database/',
#                     thumb_folder='./static/thumb_images/',
#                     suffix='',
#                     height=200,
#                     del_former_thumb=True,
#                     )
#
# data_loader = load_data(data_path='./static/image_database/',
#                         batch_size=2,
#                         shuffle=False,
#                         transform='default',
#                         )

# Prepare model.
model_paris = code_base.getModel(weights_file="./static/weights/paris_final.pth")
#model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)

# Extract database features.
#gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1700,800)

        self.label_1 = QtWidgets.QLabel(Form)
        self.label_1.setGeometry(QtCore.QRect(130,160,351,251))
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(500, 160, 351, 251))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(900, 160, 351, 251))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(1300, 160, 351, 251))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(500, 480, 351, 251))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(900, 480, 351, 251))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(1300, 480, 351, 251))
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(600, 420, 100, 50))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(1000, 420, 100, 50))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(1400, 420, 100, 50))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(600, 740, 100, 50))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setGeometry(QtCore.QRect(1000, 740, 100, 50))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setGeometry(QtCore.QRect(1400, 740, 100, 50))
        self.label_13.setObjectName("label_13")

        self.btn_1 = QtWidgets.QPushButton(Form)
        self.btn_1.setGeometry(QtCore.QRect(40,80,140,28))
        self.btn_1.setObjectName("btn_2")
        self.btn_2 = QtWidgets.QPushButton(Form)
        self.btn_2.setGeometry(QtCore.QRect(40,120,140,28))
        self.btn_2.setObjectName("btn_2")

        self.retranslateUi(Form)
        self.btn_1.clicked.connect(self.slot_open_image)
        self.btn_2.clicked.connect(self.retrieval_image)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self,Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Image Search"))
        self.label_1.setText(_translate("Form","TextLabel"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        self.label_3.setText(_translate("Form", "TextLabel"))
        self.label_4.setText(_translate("Form", "TextLabel"))
        self.label_5.setText(_translate("Form", "TextLabel"))
        self.label_6.setText(_translate("Form", "TextLabel"))
        self.label_7.setText(_translate("Form", "TextLabel"))

        self.btn_1.setText(_translate("Form","Choose a picture"))
        self.btn_2.setText(_translate("Form", "Search"))


class window(QtWidgets.QMainWindow,Ui_Form):
    def __init__(self):
        super(window,self).__init__()
        self.cwd = os.getcwd()
        self.setupUi(self)
        self.labels = [self.label_2, self.label_3, self.label_4, self.label_5, self.label_6, self.label_7]
    def slot_open_image(self):
        self.files, filetype = QFileDialog.getOpenFileName(self, 'open an image',self.cwd, "*.jpg, *.png, *,JPG, *.JPEG, All Files(*")
        self.jpg = QtGui.QPixmap(self.files).scaled(self.labels[0].width(), self.labels[0].height())
        self.label_1.setPixmap(self.jpg)

    def retrieval_image(self):
        start_time = time.time()
        # Query.
        # query_image = load_query_image(self.files)
        # # Extract query features.
        # query_feature = extract_feature_query(model=model, img=query_image)
        # # Sort.
        # similarity, index = sort_img(query_feature, gallery_feature)
        # sorted_paths = [image_paths[i] for i in index]
        # tmb_images = ['./static/thumb_images/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
        (path, filename) = os.path.split(self.files)
        print(filename)
        filename = '/static/data/paris/images/' + filename
        tmb_images,similarity = code_base.inference_on_single_labelled_image_pca_web_original(model_paris, filename)
        tmb_images = ['.' + tmb_image for tmb_image in tmb_images]
        for i in range(6):
            output_image = QtGui.QPixmap(tmb_images[i]).scaled(self.labels[i].width(), self.labels[i].height())
            self.labels[i].setPixmap(output_image)
            a = i+8
        self.label_8.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[0])))
        self.label_9.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[1])))
        self.label_10.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[2])))
        self.label_11.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[3])))
        self.label_12.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[4])))
        self.label_13.setText(QtCore.QCoreApplication.translate("Form", "similarity: " + str(similarity[5])))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my = window()
    my.show()
    sys.exit(app.exec_())



