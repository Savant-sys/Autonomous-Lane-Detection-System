import cv2
import numpy as np  

# vidcap = cv2.VideoCapture('/Users/michaelkhuri/Desktop/Autonomous-Lane-Detection-System/videos/testdashcam.mov')
vidcap = cv2.VideoCapture('/Users/michaelkhuri/Desktop/Autonomous-Lane-Detection-System/videos/LaneVideo.mp4')
success, image = vidcap.read()

while success:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640, 480))

    # Define the four points for the trapezoidal region for perspective transform
    tl = (222, 387)  # Top-left corner of the rectangle
    bl = (70, 472)  # Bottom-left corner of the rectangle
    tr = (400, 380)  # Top-right corner of the rectangle
    br = (538, 472)  # Bottom-right corner of the rectangle

    cv2.circle(frame, tl, 5, (0, 255, 0), -1)
    cv2.circle(frame, bl, 5, (0, 255, 0), -1)
    cv2.circle(frame, tr, 5, (0, 255, 0), -1)
    cv2.circle(frame, br, 5, (0, 255, 0), -1)

    # applying perspective transform
    pst1 = np.float32([tl, bl, tr, br])
    pst2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])    

    # matrix to wrap the image for bird eye view
    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    ### Object Detection ###
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # move windows (for macOS)
    cv2.moveWindow("Original", 0, -100)
    cv2.moveWindow("Transformed", 650, -100)
    cv2.moveWindow("Lane Detection - Image Thresholding", 800, 250)

    # Display the frames
    cv2.imshow("Original", frame)
    cv2.imshow("Transformed", transformed_frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    