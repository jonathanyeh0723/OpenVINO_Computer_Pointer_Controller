import cv2
import numpy as np
import logging as log

from openvino.inference_engine import IECore

class FaceDetectionModel:
    
    def __init__(self, model_name, device = "CPU", extensions = None):
    

        self.model_name = model_name
        self.device = device
        self.extensions = extensions

        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0] + ".bin"

        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        
    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model = self.model_structure, weights = self.model_weights)

        # Check for supported layers 
        supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)

        # Check for any unsupported layers, and let the user know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        self.exec_network = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)

        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape

    def predict(self, image, prob_threshold):

        img_preprocessed = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_blob: img_preprocessed})

        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords) == 0):
            return 0, 0

        coords = coords[0]
        w = image.shape[1]
        h = image.shape[0]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face, coords

    def check_model(self):
        ''

    def preprocess_input(self, image):

        n, c, h, w = self.input_shape

        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape((n, c, h, w))
        
        return p_frame

    def preprocess_output(self, outputs, prob_threshold):

        coords = []
        result = outputs[self.output_blob][0][0]
        for obj in result:
            conf = obj[2]
            if conf >= prob_threshold:
                xmin = obj[3]
                ymin = obj[4]
                xmax = obj[5]
                ymax = obj[6]

                coords.append([xmin, ymin, xmax, ymax])

        return coords
