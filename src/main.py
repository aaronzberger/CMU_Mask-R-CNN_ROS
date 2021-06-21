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
from geometry_msgs.msg import Pose2D
from cv_converter import CV_Converter
from std_msgs.msg import Header
import numpy as np
import time


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
        if not data.header.seq % self.cfg['use_image_every'] == 0:
            return
        cv_img = self.cv_converter.msg_to_cv(data)

        # If a vertical flip is needed
        cv_img = cv.flip(cv_img, flipCode=-1)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

        scores, bboxes, masks, output = self.model.forward(cv_img)
        masks = masks.astype(np.uint8) * 255

        # If you wish to publish the visualized results
        visualized = self.model.visualize(cv_img, output)
        self.pub_results.publish(self.cv_converter.cv_to_msg(cv.flip(visualized, flipCode=-1)))

        # Convert outputs from forward call to predictions message
        pub_msg = predictions()
        pub_msg.header = Header(stamp=data.header.stamp)

        pub_msg.scores = scores
        pub_msg.bboxes = [BoundingBox2D(center=Pose2D(x=i[0], y=i[1]), size_x=i[2], size_y=i[3]) for i in bboxes]
        pub_msg.masks = [self.cv_converter.cv_to_msg(masks[i], mono=True) for i in range(masks.shape[0])]

        pub_msg.source_image = self.cv_converter.cv_to_msg(cv.flip(cv_img, flipCode=-1))

        self.pub_predictions.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('cnn', log_level=rospy.INFO)

    cnn_node = CNN_Node()

    rospy.loginfo('started cnn node')

    rospy.spin()
