import cv2
import pickle
import cvzone
import numpy as np

PRACTICE_MODE = False #Infinite loop

VIDEO_DIR = "./dataset/carPark.mp4"
OUTPUT_DIR = "./output.mp4"
PARKING_SPACE_POSITIONS_LIST_DIR = "./ParkingSpacePosList.pkl" #pickle
WIDTH, HEIGHT = 107, 48     #parking space width, height

#Video feed
cap = cv2.VideoCapture("./dataset/carPark.mp4")

#Load parking space position list [(x, y), ...]
with open(PARKING_SPACE_POSITIONS_LIST_DIR, "rb") as f:
    ParkingSpacePosList = pickle.load(f)

#Set for output
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter(OUTPUT_DIR, fourcc, fps, (w, h))


def checkParkingSpace(ProcessedImg):

    #Set initial Unoccupied parking space count
    spaceCounter = 0
    
    for pos in ParkingSpacePosList:
        assert len(pos) == 2
        x, y = pos
        
        #Crop image (only parking space) & Count black dot
        imgCrop = ProcessedImg[y : y + HEIGHT, x : x + WIDTH]
        #cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)
        #cvzone.putTextRect(img, str(count), (x, y + HEIGHT - 5), scale = 1, thickness = 2, offset = 0, colorR = (0, 0, 255))
        
        #Dot threshold: 900
        if count < 900:
            color = (0, 255, 0) #Green color
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255) #Red color
            thickness = 2
            
        
        cv2.rectangle(img, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), color, thickness)
        #cvzone.putTextRect(img, str(count), (x, y + HEIGHT - 5), scale = 1, thickness = 2, offset = 0, colorR = color)
        
    cvzone.putTextRect(img, f"Free: {spaceCounter}/{len(ParkingSpacePosList)}", (100, 50), scale = 3, thickness = 5, offset = 20, colorR = (0, 0, 0))

while True:
    
    #PRACTICE_MODE = True >> Initial loop
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        if PRACTICE_MODE == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            break
    
    #Image preprocessing    
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color image >> gray image
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1) #image blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16) #Gray blur image >> binary image (white, black)
    imgMedian = cv2.medianBlur(imgThreshold, 5) #Remove salt-pepper noise
    
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations = 1) #Expand image (dependent on kernel)
    
    #Count parking space
    checkParkingSpace(imgDilate)

    #Confirm
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    #Output
    out.write(img)

out.release()
cv2.destroyAllWindows()