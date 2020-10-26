import cv2
import os
import logging
import numpy as np
import argparse
import time

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel

from mouse_controller import MouseController
from input_feeder import InputFeeder

def get_args():

    # Create parser object
    parser = argparse.ArgumentParser()
    
    # Createe the arguments
    parser.add_argument("-fd", "--facedetectionmodel", required = True, type = str, help = "Required. Path to .xml file with Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required = True, type = str, help = "Required. Path to .xml file with Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required = True, type = str, help = "Required. Path to .xml file with Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimationmodel", required = True, type = str, help = "Required. Path to .xml file with Gaze Estimation model.")
    parser.add_argument("-i", "--input", required = True, type = str, help = "Required. To specify input video or webcam")
    
    parser.add_argument("-d", "--device", default= "CPU", type = str, help = "To specify target device for inference")
    parser.add_argument("-l", "--cpu_extension", required = False, type = str, default = None, help = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-pt", "--prob_threshold", default = 0.6, type = float, help = "Probability threshold for detections filtering")

    parser.add_argument("-s", "--show", required = False, nargs = '+', default = [],
            help = "To show the inference results. Use example: -s fd fld hp ge; "
                             "fd for Face Detection, fld for Facial Landmark Detection, "
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
   
    args = parser.parse_args()

    return args

def main():

    args = get_args()
    checkFlags = args.show
    
    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None
   
    load_start = time.time()

    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video", inputFilePath)
    
    modelPathDict = {"FaceDetectionModel": args.facedetectionmodel, "FacialLandmarksDetectionModel": args.faciallandmarkmodel, "GazeEstimationModel": args.gazeestimationmodel, "HeadPoseEstimationModel": args.headposemodel}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified " + fileNameKey + " xml file" )
            exit(1)
            
    fdm = FaceDetectionModel(modelPathDict["FaceDetectionModel"], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict["FacialLandmarksDetectionModel"], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict["GazeEstimationModel"], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict["HeadPoseEstimationModel"], args.device, args.cpu_extension)
    
    mc = MouseController("medium", "fast")
    
    inputFeeder.load_data()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()

    # Model load time
    model_load_time = time.time() - load_start 
    

    frame_count = 0
    inf_start = time.time()

    for flag, frame in inputFeeder.next_batch():
        if not flag:
            break
        frame_count += 1

        if frame_count%5 == 0:
            cv2.imshow("video", frame)
    
        key_pressed = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        
        if type(croppedFace) == int:
            logger.error("Unable to detect the face.")
            
            if key_pressed == 27:
                break

            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        coords, gaze_vector = gem.predict(left_eye, right_eye, hp_out)

        # Inference time
        inference_time = time.time() - inf_start

        if (not len(checkFlags) == 0):
            check_frame = frame.copy()
            if "fd" in checkFlags:
                check_frame = croppedFace
            
            if "fld" in checkFlags:
                cv2.rectangle(croppedFace, (eye_coords[0][0] - 20, eye_coords[0][1] - 20), (eye_coords[0][2] + 20, eye_coords[0][3] + 20), (255, 10, 10), 2)
                cv2.rectangle(croppedFace, (eye_coords[1][0] - 20, eye_coords[1][1] - 20), (eye_coords[1][2] + 20, eye_coords[1][3] + 20), (255, 10, 10), 2)
            
            '''    
            if "hp" in checkFlags:
                cv2.putText(check_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,, 0, 0), 1)
            '''

            if "ge" in checkFlags:
                x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (0, 255, 0), 3)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (0, 255, 0), 3)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (0 ,255, 0), 3)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (0, 255, 0), 3)

                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                
            cv2.imshow("Frames", cv2.resize(check_frame, (480, 480)))
        
        if frame_count%5 == 0:
            mc.move(coords[0], coords[1])    
        
        if key_pressed == 27:
                break

    # FPS
    fps = frame_count / inference_time

    logger.error("[ INFO ] Executed demo successfully.")
   
    # Check the time to load the model, perform inference, and FPS
    logger.error("[ INFO ] Model loading time: {:.3f} seconds".format(model_load_time))
    logger.error("[ INFO ] Inference time: {:.3f} seconds".format(inference_time))
    logger.error("[ INFO ] FPS: {:.3f} seconds".format(fps))

    cv2.destroyAllWindows()
    inputFeeder.close()
     
if __name__ == "__main__":
    main() 
