import cv2
import numpy as np

# 0) Open the video
cap = cv2.VideoCapture("testdashcam.mov")

# Safety check
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()


def region_of_interest(img):
    # img is a single-channel (grayscale/edges) image
    height, width = img.shape

    # Black image (same size)
    mask = np.zeros_like(img)

    # Polygon that covers the road area (bottom trapezoid)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.65)),
        (int(width * 0.45), int(height * 0.65))
    ]], np.int32)

    # Fill the polygon with white on the mask
    cv2.fillPoly(mask, polygon, 255)

    # Keep only the region inside the polygon
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask


def filter_colors(frame):
    # Convert to HLS (better for separating light/shadow)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # White lane mask
    lower_white = np.array([0, 135, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # Yellow lane mask
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    # Combine both masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply mask to original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def make_coordinates(image, line_params):
    slope, intercept = line_params
    height = image.shape[0]

    # y1 = bottom of the image, y2 = some height up (like horizon-ish)
    y1 = height
    y2 = int(height * 0.7)

    # x = (y - b) / m  (rearranged y = mx + b)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fits = []
    right_fits = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue  # vertical line, skip

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # negative slope = left lane (in typical dashcam view)
        if slope < -0.65:
            left_fits.append((slope, intercept))
        # positive slope = right lane
        elif slope > 0.65:
            right_fits.append((slope, intercept))

    lane_lines = []

    if left_fits:
        left_avg = np.mean(left_fits, axis=0)
        lane_lines.append(make_coordinates(image, left_avg))

    if right_fits:
        right_avg = np.mean(right_fits, axis=0)
        lane_lines.append(make_coordinates(image, right_avg))

    return lane_lines

while True:
    # 1) Read a frame
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # 2) Color filter first (keep mostly lanes)
    filtered = filter_colors(frame)

    # 3) Grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # 4) Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5) Edges
    edges = cv2.Canny(blur, 50, 150)

    # 6) Region of interest
    cropped, roi_mask = region_of_interest(edges)

    # 7) Hough Lines
    lines = cv2.HoughLinesP(
        cropped,
        rho=.95,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=15,
        maxLineGap=100
    )

    # 8) Draw averaged left/right lanes
    line_image = np.zeros_like(frame)

    if lines is not None:
        # # draw all lines (for debugging)
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]

        #     # Avoid vertical divide-by-zero
        #     if x2 == x1:
        #         continue

        #     slope = (y2 - y1) / (x2 - x1)

        #     # Only draw if it looks like a lane (angled line)
        #     if abs(slope) > 0.5:
        #         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # # Draw averaged lanes
        averaged_lines = average_slope_intercept(frame, lines)

        for x1, y1, x2, y2 in averaged_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 9) Overlay lines on original frame
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cropped_combo = cv2.bitwise_and(combo, combo, mask=roi_mask)


    # 10) Show windows # for macOS
    cv2.moveWindow("Filtered", 800, -100)
    cv2.moveWindow("Edges Only", 800, 350)
    cv2.moveWindow("Lane Detection", 0, -100)
    cv2.moveWindow("Cropped Lane Detection", 0, 350)

    cv2.imshow("Filtered", filtered)
    cv2.imshow("Edges Only", cropped)
    cv2.imshow("Lane Detection", combo)
    cv2.imshow("Cropped Lane Detection", cropped_combo)
    

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
