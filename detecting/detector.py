import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils

# load haarcascade xml file (put in project folder OR give path)
face_cascade = cv2.CascadeClassifier('detecting/haarcascade_frontalface_alt.xml')

# detect face in a rectangle
def DetectFace(img, cascade = face_cascade, minimumFeatureSize=(20,20)):
    if cascade.empty():
        raise(Exception("Problem Loading Haarcascade XML file"))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    
    # if returned array is 0
    if len(rects)==0:
        return[]
    
    # convert last coord from (width,height) to (maxX, maxY)
    rects[:,2:] += rects[:,:2]
    
    return rects

predictor = dlib.shape_predictor("detecting/shape_predictor_68_face_landmarks.dat")

# making function for cropping eyes
def CropEyes(frame):
    
    # defining gray color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect the face as a grayscale image
    te = DetectFace(gray, minimumFeatureSize=(80, 80))
    
    # return none if no face detected
    # return the bigger one if more then one face detected
    # if only one face detected, make it one dimension
    
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te
        
    # keep the facial regions of the whole frame
    face_rect = dlib.rectangle(left= int(face[0]), top = int(face[1]),  right = int(face[2]), bottom = int(face[3]))

    # determining the facial landmarks
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    # picking the facial landmark indexes of left and right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    
    # highlighting the eyes
    leftEyeHull = cv2.convexHull(leftEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
    # computing the height of the left and right eyes
    l_uppery = min(leftEye[1:3,1])
    l_lowy = max(leftEye[4:,1])
    l_dify = abs(l_uppery - l_lowy)
    
    r_uppery = min(rightEye[1:3,1])
    r_lowy = max(rightEye[4:,1])
    r_dify = abs(r_uppery - r_lowy)
    
    # computing width of eye
    lw = (leftEye[3][0] - leftEye[0][0])
    rw = (rightEye[3][0] - rightEye[0][0])
    
    # we want the image for the cnn to be (26,34)
    # so we add the half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    minxl = (leftEye[0][0] - ((34-lw)/2))
    maxxl = (leftEye[3][0] + ((34-lw)/2)) 
    minyl = (l_uppery - ((26-l_dify)/2))
    maxyl = (l_lowy + ((26-l_dify)/2))
    
    minxr = (rightEye[0][0]-((34-rw)/2))
    maxxr = (rightEye[3][0] + ((34-rw)/2))
    minyr = (r_uppery - ((26-r_dify)/2))
    maxyr = (r_lowy + ((26-r_dify)/2))
    
    # crop the eyes rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]
    
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3],  right_eye_rect[0]:right_eye_rect[2]]
        
    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    
    # resize for the conv net
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)
    
    # return left and right eye
    return left_eye_image, right_eye_image


# making the input image the same format/data-type as trained images
def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img

def main():

    # opening camera 
    camera = cv2.VideoCapture(0)
    
    # loading the model we have trained
    model = load_model('training/trainedBlinkModel.hdf5')
    
    # blinks is the number of total blinks
    # close_counter is the counter for consecutive close predictions
    # mem_counter the counter of the previous loop 
    # close_counter = left_blinks = right_blinks = double_blinks = mem_counter= 0
    double_close_counter = double_blinks = double_mem_counter = 0
    left_close_counter = left_blinks = left_mem_counter = 0
    right_close_counter = right_blinks = right_mem_counter = 0
    
    both_eye_state = 'open'
    left_eye_state = 'open'
    right_eye_state = 'open'
    
    # counters for counting the open state
    double_open_counter = 0
    left_open_counter = 0
    right_open_counter = 0

    while True:
        
        ret, frame = camera.read()
        
        # detect eyes
        eyes = CropEyes(frame)
        if eyes is None:
            continue
        else:
            left_eye,right_eye = eyes
        
        # average the predictions of the two eyes 
        left_eye_prediction = model.predict(cnnPreprocess(left_eye))
        right_eye_prediction = model.predict(cnnPreprocess(right_eye)) 
        both_eye_prediction = (left_eye_prediction + right_eye_prediction)/2.0
        
        # blinks
        # if both the eyes are open reset the counter for close eyes
        if both_eye_prediction > 0.5:
            both_eye_state = 'open'
            double_close_counter = 0
            double_open_counter += 1
        else:
            both_eye_state = 'close'
            double_close_counter += 1
        
        # if the lefteye is open reset the counter for close eyes
        if left_eye_prediction > 0.5:
            left_eye_state = 'open'
            left_close_counter = 0
            left_open_counter += 1
        else:
            left_eye_state = 'close'
            left_close_counter += 1
        
        # if the right eye is open reset the counter for close eyes
        if right_eye_prediction > 0.5:
            right_eye_state = 'open'
            right_close_counter = 0
            right_open_counter += 1
        else:
            right_eye_state = 'close'
            right_close_counter += 1
            
        # if both eyes were open in one frame and then closed in two consecutive frames
        # open -> close -> close
        # count a double blink
        if double_open_counter > 1 and double_close_counter > 2:
            double_blinks += 1
            double_open_counter = 0
            double_close_counter = 0
        
        
        # if left eye was open in one frame and then closed in two consecutive frames
        # open -> close -> close
        # count a left blink
        elif left_open_counter > 1 and left_close_counter > 2:
            left_blinks += 1
            left_open_counter = 0
            left_close_counter = 0
        
        # if right eye was open in one frame and then closed in two consecutive frames
        # open -> close -> close
        # count a right blink
        elif right_open_counter > 1 and right_close_counter > 2:
            right_blinks += 1
            right_open_counter = 0
            right_close_counter = 0

        # keep the counter for the next loop 
        double_mem_counter = double_close_counter   
        left_mem_counter = left_close_counter   
        right_mem_counter = right_close_counter
        
        # mem_counter = close_counter
        # draw the total number of blinks on the frame
        cv2.putText(frame, "Double Blinks: {}".format(double_blinks), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
        cv2.putText(frame, "Left Blinks: {}".format(left_blinks), (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        cv2.putText(frame, "Right Blinks: {}".format(right_blinks), (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # show the frame
        cv2.imshow('Blink Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    del(camera)
    


    
if __name__ == "__main__":
    main()