#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

import os

import rospy
from sensor_msgs.msg import Image
from model import Mask_R_CNN
import json
from CMU_Mask_R_CNN.msg import predictions
import cv2 as cv
from vision_msgs.msg import BoundingBox2D
from cv_converter import CV_Converter
from std_msgs.msg import Header


class CNN_Node:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'config.json')

        with open(path) as file:
            self.cfg = json.load(file)

        self.model = Mask_R_CNN(self.cfg)

        self.cv_converter = CV_Converter()

        # Initialize the subscribers last or else the callback will trigger
        # when the model hasn't been created
        self.sub_rectified = rospy.Subscriber(
            '/mapping/left/image_raw', Image, self.image_callback)
        self.pub_predictions = rospy.Publisher(
            '/cnn_predictions', predictions, queue_size=1)
        self.pub_results = rospy.Publisher(
            '/cnn_vis', Image, queue_size=1)

    def image_callback(self, data):
        cv_img = self.cv_converter.msg_to_cv(data)

        # If a vertical flip is needed
        cv_img = cv.flip(cv_img, flipCode=0)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

        cv.imwrite('/home/aaron/testimage.png', cv_img)

        scores, bboxes, masks, output = self.model.forward(cv_img)

        # If you wish to publish the visualized results
        visualized = self.model.visualize(cv_img, output)
        self.pub_results.publish(self.cv_converter.cv_to_msg(visualized))

        # TODO: Convert outputs from forward call to predictions message
        pub_msg = predictions()
        pub_msg.header = Header(stamp=rospy.Time.now())

        self.pub_predictions.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('cnn', log_level=rospy.INFO)

    cnn_node = CNN_Node()

    rospy.loginfo('started cnn node')

    rospy.spin()
