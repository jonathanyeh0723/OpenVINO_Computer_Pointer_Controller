import cv2
import numpy as np
from openvino.inference_engine import IECore

class FacialLandmarksDetectionModel:

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

        self.exec_network = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)

        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        
    def predict(self, image):
        
        img_preprocessed = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_blob: img_preprocessed})

        coords = self.preprocess_output(outputs)
        w = image.shape[1]
        h = image.shape[0]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        left_xmin = coords[0] - 20
        left_ymin = coords[1] - 20
        left_xmax = coords[0] + 20
        left_ymax = coords[1] + 20

        right_xmin = coords[2] - 20
        right_ymin = coords[3] - 20
        right_xmax = coords[2] + 20
        right_ymax = coords[3] + 20

        left_eye = image[left_ymin: left_ymax, left_xmin: left_xmax]
        right_eye = image[right_ymin: right_ymax, right_xmin: right_xmax]
        eye_coords = [[left_xmin, left_ymin, left_xmax, left_ymax], [right_xmin, right_ymin, right_xmax, right_ymax]]

        return left_eye, right_eye, eye_coords
        
    def check_model(self):

        # Check for supported layers 
        supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)

        # Check for any unsupported layers, and let the user know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):

        n, c, h, w = self.input_shape

        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        p_frame = cv2.resize(image_cvt, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape((n, c, h, w))

        return p_frame

    def preprocess_output(self, outputs):

        outs = outputs[self.output_blob][0]

        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)
