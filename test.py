from shapely import geometry
import cv2 as cv
import numpy as np
import cupy as cp
from random import randint

max_images = 6

block_size = 21

kroket = 1
bamischijf = 2
frikandel = 3
kaassoufle = 4
kipcorn = 5

snacklist = [
    ((30, 48), (26, 42), kroket), 
    ((18, 33), (15, 25), frikandel), 
    ((80, 99), (37, 57), bamischijf),
    ((58,80), (22,37), kaassoufle),
    ((12,28), (22,37), kipcorn)
    ]

def main():
    cv.namedWindow("krokante filter", cv.WINDOW_NORMAL)
    cv.resizeWindow("krokante filter", 800, 800)

    create_trackbars()

    cam = cv.VideoCapture(1)

    # _, orig = cam.read()

    # orig = cv.imread("../pics/test1.jpg")

    data_list = gen_data(snacklist)

    image_index = 0
    win = 0
    while(cam.isOpened()):

        _, orig = cam.read()

        key = cv.waitKey(50)

        cv.cvtColor(orig, cv.COLOR_BGR2RGB)

        filtered = mask_frituur(orig)

        gray_masked = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)

        thresh = cv.threshold(gray_masked, 20, 255, cv.THRESH_BINARY)[1]

        black = orig.copy()

        black.fill(0)

        shapes = get_shapes(thresh)

        mask = cv.cvtColor(cv.drawContours(black, shapes, -1, (255,255,255), -1)  , cv.COLOR_RGB2GRAY)
        masked = cv.bitwise_and(orig, orig, mask=mask)

        dis = cv.drawContours(orig.copy(), shapes, -1, (0, 255, 0), 3)

        stages = [orig, dis, filtered, thresh]

        for shape in shapes:
            rotated_rect = cv.minAreaRect(shape)
            shape = np.squeeze(shape)

            cropped = crop_minAreaRect(masked, rotated_rect)
            cropped_mask = crop_minAreaRect(mask, rotated_rect)

            blurred_cropped = cv.blur(cropped, (3, 3))

            cropped_canny = cv.Canny(blurred_cropped, cv.getTrackbarPos("canny th1", "krokante filter"), cv.getTrackbarPos("canny th2", "krokante filter"))

            cropped_dilated = cv.dilate(cropped_canny, np.ones((cv.getTrackbarPos("dilation", "krokante filter"), cv.getTrackbarPos("dilation", "krokante filter")), np.float32))

            stages.append(cropped_dilated)
            
            print(cv.mean(cropped, mask=cropped_mask))

            if key == 98:
                print_shape_info(cropped_dilated, shape)
                ratios = get_shape_info(cropped_dilated, shape)
                print(KNN(data_list, ratios[1] * 100, ratios[0] * 100, 5))

        if win >= len(stages):
            win = len(stages) - 1

        cv.imshow("krokante filter", stages[win])

        if key == 100 and win < len(stages) - 1:
            win += 1
        elif key == 97 and win > 0:
            win -= 1
        # elif key == 119 and image_index < max_images - 1: 
        #     image_index += 1
            
        #     string = "../pics/test" + str(image_index + 1) + ".jpg"

        #     orig = cv.imread(string)

        # elif key == 115 and image_index > 0:
        #     image_index -= 1

        #     string = "../pics/test" + str(image_index + 1) + ".jpg"

        #     orig = cv.imread(string)

        elif key == 27:
            break        
        
def nothing(x):
    pass        

def create_trackbars():
    cv.createTrackbar("range", "krokante filter", 0, 255, nothing)

    cv.createTrackbar("canny th1", "krokante filter", 60, 1000, nothing)
    cv.createTrackbar("canny th2", "krokante filter", 0, 1000, nothing)

    cv.createTrackbar("dilation", "krokante filter", 2, 100, nothing)

    
def get_shape_info(orig, poly):
    white = cv.countNonZero(orig)
    black = cv.contourArea(poly) - white

    bw_ratio = white / black
    
    # if bw_ratio > 1.0:
    #     bw_ratio =  1.0 / bw_ratio
        
    wh_ratio = orig.shape[1] / orig.shape[0]

    if wh_ratio > 1.0:
        wh_ratio = 1.0 / wh_ratio

    return (white / black, orig.shape[1] / orig.shape[0])
    
def print_shape_info(orig, poly):
    white = cv.countNonZero(orig)
    black = cv.contourArea(poly) - white

    bw_ratio = white / black
    
    # if bw_ratio > 1.0:
    #     bw_ratio =  1.0 / bw_ratio

    wh_ratio = orig.shape[1] / orig.shape[0]

    if wh_ratio > 1.0:
        wh_ratio = 1.0 / wh_ratio

    print("width: " + str(orig.shape[1]))
    print("height: " + str(orig.shape[0]))
    print("wh ratio: " + str(wh_ratio))

    print("white: " + str(white))
    print("black: " + str(black))
    print("bw ratio: " + str(bw_ratio))

def mask_frituur(image):

    blur_filter = np.ones((4, 4), np.float32) * 0.28571428571
    dilation_filter = np.ones((10, 10), np.float32)

    blurred  = cv.filter2D(image, -1, blur_filter)
    filtered = cv.bitwise_not(filter_white(blurred))

    dilated = cv.dilate(filtered, dilation_filter)

    return cv.bitwise_and(image, image, mask=dilated)

def crop_minAreaRect(orig, rect):
    mult = 1

    box = cv.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))

    M = cv.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv.getRectSubPix(orig, size, center)
    cropped = cv.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    return cv.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2)).copy()

def filter_white(image):
    block_size = cv.getTrackbarPos("range", "krokante filter")

    if(block_size < 3):
        block_size = 3
    elif(block_size % 2 == 0):
        block_size = block_size + 1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block_size, 0)
    
    return mask.copy()

def get_shapes(img):
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    shapes = []

    # loop through the contours
    for i,cnt in enumerate(cnts):
        # if the contour has no other contours inside of it
        if hier[0][i][2] == -1 :
            # if the size of the contour is greater than a threshold
            if  cv.contourArea(cnt) > 100000:
                shapes.append(cnt)
    
    return shapes

def average_color(img, mask):
    return img.mean(mask=mask)

def gen_data(snacklist):
    datalist = []
    for snack in snacklist:
        for i in range(0,25):
            LB = randint(snack[0][0], snack[0][1])
            KO = randint(snack[1][0], snack[1][1])
            snack2d = [LB,KO,snack[2]]
            datalist.append(snack2d)
    return datalist

def KNN(biglist, LBV, KOV,k):
    dislist = []
    predictlist = []
    prediction = 0
    counter = 0

    if (k % 2 == 0):
        k = k + 1

    for i in biglist:
        tempi = i
        dis = np.sqrt(abs(i[0]-LBV)**2+abs(i[1]-KOV)**2)
        tempi.append(dis)
        dislist.append(tempi)
        dislist.sort(key=takeSecond)

    for j in dislist[0:k]:
        predictlist.append(j[2])
    
    for p in predictlist:
        freq = predictlist.count(p)
        if(freq > counter):
            counter = freq
            prediction = p
    
    return prediction

def takeSecond(elem):
    return elem[3]

if __name__ == "__main__":
    main()