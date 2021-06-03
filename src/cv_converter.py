from cv_bridge import CvBridge

class CV_Converter:
    def __init__(self):
        self.cv_bridge = CvBridge()

    def msg_to_cv(self, msg):
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cv_to_msg(self, cv_img):
        return self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")