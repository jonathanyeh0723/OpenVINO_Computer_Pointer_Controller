import cv2
import numpy as np
import math

from openvino.inference_engine import IECore

class GazeEstimationModel:
    
    def __init__(self, model_name, device = "CPU", extensions = None):

        self.model_name = model_name
        self.device = device
        self.extensions = extensions

        self.model_topology = self.model_name
        self.model_binary = self.model_name.split(".")[0] + ".bin"

        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model = self.model_topology, weights = self.model_binary)

        self.exec_network = self.plugin.load_network(self.network, device_name = self.device, num_requests = 1)

        self.input_blob = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_blob[1]].shape
        self.output_blob = [j for j in self.network.outputs.keys()]
    
    def predict(self, left_eye_image, right_eye_image, hpa):

        left_img_preprocessed, right_img_preprocessed = self.preprocess_input(left_eye_image, right_eye_image)
        outputs = self.exec_network.infer({"head_pose_angles": hpa, "left_eye_image": left_img_preprocessed, "right_eye_image": right_img_preprocessed})
        
        coords, gaze_vector = self.preprocess_output(outputs, hpa)

        return coords, gaze_vector

    def check_model(self):

        supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)


    def preprocess_input(self, left_eye, right_eye):

        n, c, h, w = self.input_shape

        # Left
        left_pframe = cv2.resize(left_eye, (w, h))
        left_pframe = left_pframe.transpose((2, 0, 1))
        left_pframe = left_pframe.reshape((n, c, h, w))

        # Right
        right_pframe = cv2.resize(right_eye, (w, h))
        right_pframe = right_pframe.transpose((2, 0, 1))
        right_pframe = right_pframe.reshape((n, c, h, w))

        return left_pframe, right_pframe

    def preprocess_output(self, outputs, hpa):

        gaze_vector = outputs[self.output_blob[0]].tolist()[0]
        
        rollValue = hpa[2] 

        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] * sinValue + gaze_vector[1] * cosValue

        return (newx, newy), gaze_vector
