import cv2
import pickle

#Set parking space size (width, height)
WIDTH, HEIGHT = 107, 48
PARKING_SPACE_POSITIONS_LIST_DIR = "./ParkingSpacePosList.pkl" #pickle
COLOR = (255, 0, 255)

#Load position list of parking spaces
try: 
    with open(PARKING_SPACE_POSITIONS_LIST_DIR, "rb") as f:
        ParkingSpacePosList = pickle.load(f)
except:
    ParkingSpacePosList = list()

#Define function (To get parking spaces list)
def Get_ParkingSpace_Using_MouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        ParkingSpacePosList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(ParkingSpacePosList):
            x1, y1 = pos
            if x1 < x < x1 + WIDTH and y1 < y < y1 + HEIGHT:
                ParkingSpacePosList.pop(i)
    
    #Save position list of parking spaces
    with open (PARKING_SPACE_POSITIONS_LIST_DIR, "wb") as f:
        pickle.dump(ParkingSpacePosList, f)

#Get position list of parking spaces
while True:
    img = cv2.imread("./dataset/carParkImg.png")
    for pos in ParkingSpacePosList:
        assert len(pos) == 2
        cv2.rectangle(img, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), COLOR, 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", Get_ParkingSpace_Using_MouseClick)
    cv2.waitKey(1)