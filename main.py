import cv2
import numpy as np

cap = cv2.VideoCapture("testdashcam.mp4")

def region_of_interest(img):
    height, width = img.shape

    mask = np.zeros_like(img)

    # Polygon that covers the road
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# this is for blocking shadow and light variations
def filter_colors(frame):
    # Convert to HLS (better for light/shadow separation)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # White color mask
    lower_white = np.array([0, 180, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result


while True:
    # ret will be True if frame is read correctly
    # frame is the actual frame
    ret, frame = cap.read()
    # basically same thing as below:
    # result = cap.read()
    # ret = result[0]
    # frame = result[1]
    if not ret:
        break

    # 1) Grayscale
    filtered = filter_colors(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # 2) Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Edges
    edges = cv2.Canny(blur, 50, 150)

    # 4) Region of interest
    cropped = region_of_interest(edges)

    # 5) Hough Lines
    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # 6) Draw lines
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid vertical divide-by-zero
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Only draw if it looks like a lane (angled line)
            if abs(slope) > 0.5:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)


    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    
    cv2.imshow("Lane Detection", combo)
    cv2.imshow("Edges Only", cropped)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
