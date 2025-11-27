import cv2
import numpy as np

def empty(a):
    pass

# Load your image (replace 'road_image.jpg' with your actual image file)
# It's important to use the image you are working with to tune correctly
img = cv2.imread('pic.png')
vid = cv2.VideoCapture('testdashcam.mov')

if img is None:
    print("Error: Image not found or could not be read.")
    exit()

if vid is None:
    print("Error: Video not found or could not be read.")
    exit()

blank = np.zeros((1, 1, 3), dtype=np.uint8)

# Create a window to hold all HLS trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for Hue, Lightness, and Saturation range limits
# H: 0-179, L: 0-255, S: 0-255 in OpenCV
cv2.createTrackbar("H Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("L Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("L Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, empty)

while True:
    ret, frame = vid.read()
    if not ret:
        break 
    
    # Get current positions of the trackbars
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    l_min = cv2.getTrackbarPos("L Min", "Trackbars")
    l_max = cv2.getTrackbarPos("L Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")

    # Define the lower and upper bounds
    lower_bound = np.array([h_min, l_min, s_min])
    upper_bound = np.array([h_max, l_max, s_max])


    # Convert the image to HLS color space
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Create the mask and apply it to the original image
    mask = cv2.inRange(hls, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.moveWindow("Trackbars", 200, 0)
    cv2.moveWindow("Original Image", 0, 250)
    cv2.moveWindow("Mask", 700, -100)
    cv2.moveWindow("Result", 700, 350)

    # Display the results
    cv2.imshow("Trackbars", blank)
    cv2.imshow("Original Image", frame)
    cv2.imshow("Mask", mask) # The mask is a binary image: white where colors are in range, black otherwise
    cv2.imshow("Result", result) # Shows only the detected colors on the original background

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('p'):
        # Pause the video when 'p' is pressed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('p'):
                break

cv2.destroyAllWindows()
