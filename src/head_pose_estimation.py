import cv2
import numpy as np

from openvino.inference_engine import IECore

class HeadPoseEstimationModel:
    
    def __init__(self, model_name, device = "CPU", extensions = None):

        self.model_name = model_name
        self.device = device
        self.extensions = extensions

        self.model_xml = self.model_name
        self.model_bin = self.model_name.split(".")[0] + ".bin"

        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model = self.model_xml, weights = self.model_bin)
        self.exec_network = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)

        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape

        self.output_blob = [i for i in self.network.outputs.keys()]

    def predict(self, image):

        img_preprocessed = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_blob: img_preprocessed})
        
        res = self.preprocess_output(outputs)

        return res
        

    def check_model(self):

        supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):

        n, c, h, w = self.input_shape

        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape((n, c, h, w))
    
        return p_frame

    def preprocess_output(self, outputs):

        results = []

        results.append(outputs['angle_y_fc'].tolist()[0][0])
        results.append(outputs['angle_p_fc'].tolist()[0][0])
        results.append(outputs['angle_r_fc'].tolist()[0][0])
        
        return results
