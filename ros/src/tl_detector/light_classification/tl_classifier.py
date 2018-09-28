from styx_msgs.msg import TrafficLight
from sklearn.linear_model import LogisticRegression
import pickle
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        detect_model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
        PATH_TO_CKPT = '/home/workspace/ProgrammingCarla/tl/' + detect_model_name + '/frozen_inference_graph.pb'
        # setup tensorflow graph
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.Session(graph=self.detection_graph, config=config)
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
        #setup light state detector
        self.hist_bin_size = 2
        self.tl_state_clf = pickle.load(open('/home/workspace/ProgrammingCarla/tl/classifier.p', 'rb'))
        TrafficLight.RED
        
    def get_localization(self, image):  
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                     feed_dict={self.image_tensor: image_expanded})
            boxes=np.squeeze(boxes)
            classes =np.squeeze(classes)
            scores = np.squeeze(scores)
        return boxes, classes, scores
    
    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)
 
    def get_histogram(self,img, hist_bin_size):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data_index = 0
        X = np.zeros((1,(hist_bin_size)*12+1))
        index = 0
        for chnl in range(4):
            if chnl < 3:
                for y_pos_ind in range(3): 
                    if y_pos_ind == 0:
                        y1 = 0
                        y2 = 10
                    if y_pos_ind == 1:
                        y1 = 10
                        y2 = 20
                    if y_pos_ind == 2:
                        y1 = 20
                        y2 = 32
                    hist, bin_edges = np.histogram(np.ravel(img[y1:y2,:,chnl]),hist_bin_size)
                    hist = hist/np.sum(hist)
                    x_strt = index*hist_bin_size
                    x_stp = (index+1)*hist_bin_size
                    X[data_index,x_strt:x_stp] = hist.reshape(1,-1)
                    index += 1
            else:
                
                for y_pos_ind in range(3): 
                    if y_pos_ind == 0:
                        y1 = 0
                        y2 = 10
                    if y_pos_ind == 1:
                        y1 = 10
                        y2 = 20
                    if y_pos_ind == 2:
                        y1 = 20
                        y2 = 32
                    hist, bin_edges = np.histogram(np.ravel(gray_image[y1:y2,:]),hist_bin_size)
                    hist = hist/np.sum(hist)
                    x_strt = index*hist_bin_size
                    x_stp = (index+1)*hist_bin_size
                    X[data_index,x_strt:x_stp] = hist.reshape(1,-1)
                    index += 1
        top_int = np.mean(gray_image[:10,:])
        cen_int = np.mean(gray_image[10:20,:])
        bot_int = np.mean(gray_image[20:32,:])
        if top_int > cen_int:
            if top_int > bot_int:
                X[0,-1] = 0.0
            else:
                X[0,-1] = 1.0
        else:
            if cen_int > bot_int:
                X[0,-1] = 0.5
            else:
                X[0,-1] = 1.0
        return X

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        boxes, classes, scores = self.get_localization(image)
        traffic_boxes = []
        dim = image.shape[0:2]
        light_found = False
        for i, cls in enumerate(classes):
            if cls == 10:
                box = boxes[i]
                box_height = box[2] - box[0]
                box_width = box[3] - box[1]
                if scores[i]>0.25 and box_height/box_width>1.5:
                    traffic_boxes.append(self.box_normal_to_pixel(boxes[i], dim))
                    light_found = True

        if light_found:
            red_votes = 0
            green_votes = 0
        
            for box in traffic_boxes:
                x = self.get_histogram(cv2.resize(image[box[0]:box[2],box[1]:box[3],:],(32,32)), self.hist_bin_size)
                pr = self.tl_state_clf.predict(x)
                if pr == 0:
                    red_votes += 1
                else:
                    green_votes += 1
            if red_votes > green_votes:
                return TrafficLight.RED
            else:
                return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
