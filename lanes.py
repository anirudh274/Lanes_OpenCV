import cv2
import numpy as np

def make_coordinates (image, line_parameters):
    slope, intercept  =line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def  average_slope_intercept(image,lines):
    left_fit = [] # will contain co-ords of left lines
    right_fit = [] # will contain co-ords of right lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1) #polyfit will fit the 1st degree polynomial which would be linear function of y = mx+b and later it will return a vector
        slope  = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept)) #we will append and the tuple
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # we are importing a libaray "cvtColor" so that grayscale gets to another color
    blur = cv2.GaussianBlur(gray, (5,5), 0) #5x5 Grid
    canny = cv2.Canny(blur, 50,150)
    return canny

def displays_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2  in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) #(image, the points thresholds, the color of line, the thickness)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100,height), (550,250)]
    ]) #it will declare as numpy array
    mask  = np.zeros_like(image)
    cv2.fillPoly(mask,polygons, 255) #image, color as 255
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


################## I M A G E #########################
# image = cv2.imread('test_image.jpeg') #we are directing imread to  load the test_image
# lane_image = np.copy(image) #Thus copying our array into a new variable
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #(image, pixels, degree comparision, threshold) # we took 1 degree which is pi/180rad
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = displays_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8,line_image, 1, 1) #0.8 * line_image array elements , 2nd input array, scalar value not that much useful to define
# cv2.imshow("result", combo_image)
# cv2.waitKey(0) #this function displays the image for specified amount of milliseconds


################ V I D E O ###########################
cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read() #_ is the boolean
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #(image, pixels, degree comparision, threshold) # we took 1 degree which is pi/180rad
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = displays_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8,line_image, 1, 1) #0.8 * line_image array elements , 2nd input array, scalar value not that much useful to define
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'):  #if its 0 then we will wait for infinte time at each frame which in reality means freeze and now we will give a time duration of 1 milliseconds
        break
cap.release()
cv2.destroyAllWindows()
