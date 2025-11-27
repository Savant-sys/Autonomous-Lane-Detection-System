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
        (0, height * .94),
        (width, height * .94),
        (int(width * 0.55), int(height * 0.68)),
        (int(width * 0.45), int(height * 0.68))
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
    lower_white = np.array([0, 130, 0])
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

def fit_lane_curve(points, image):
    """
    points: list of (x, y) tuples from Hough lines for one side (left or right)
    image:  frame, used to know height
    returns: array of points to draw as a smooth curve, or None
    """
    if len(points) < 5:
        return None  # not enough data to fit a curve

    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Fit x as a function of y: x = a*y^2 + b*y + c
    a, b, c = np.polyfit(ys, xs, 2)

    height = image.shape[0]
    y_bottom = height        # bottom of the image
    y_top = int(height * 0.73)   # how high the curve goes

    # Generate many y values between bottom and top
    y_values = np.linspace(y_bottom, y_top, num=30)

    # Compute matching x for each y using the polynomial
    x_values = a * (y_values ** 2) + b * y_values + c

    # Stack into shape (N, 1, 2) for cv2.polylines
    curve_points = np.stack((x_values, y_values), axis=1).astype(np.int32)
    curve_points = curve_points.reshape((-1, 1, 2))

    return curve_points

# Initial empty point lists for preventing freaky lines, saves last good curves
## If this frame doesn’t have enough good lane data, keep using the last stable curve instead of jumping somewhere random basically
prev_left_curve = None
prev_right_curve = None

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
    edges = cv2.Canny(blur, 45, 150)

    # 6) Region of interest
    cropped, roi_mask = region_of_interest(edges)

    # 7) Hough Lines
    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=10,
        maxLineGap=200
    )

    # 8) Collect left/right lane points
    left_points = []
    right_points = []
    height, width = frame.shape[:2]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid vertical divide-by-zero
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Only consider lines that look like lanes
            if slope < -0.7 and x1 < width * 0.5 and x2 < width * 0.5:   # left lane (negative slope)
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            elif slope > 0.7 and x1 > width * 0.5 and x2 > width * 0.5:  # right lane (positive slope)
                right_points.append((x1, y1))
                right_points.append((x2, y2))
        
    # 9) Fit curves
    line_image = np.zeros_like(frame)

    left_curve = fit_lane_curve(left_points, frame)
    right_curve = fit_lane_curve(right_points, frame)

    if left_curve is not None:
        prev_left_curve = left_curve
    elif prev_left_curve is not None:
        left_curve = prev_left_curve

    if right_curve is not None:
        prev_right_curve = right_curve
    elif prev_right_curve is not None:
        right_curve = prev_right_curve

    if left_curve is not None:
        cv2.polylines(line_image, [left_curve], isClosed=False, color=(0, 255, 0), thickness=2)
    if right_curve is not None:
        cv2.polylines(line_image, [right_curve], isClosed=False, color=(0, 255, 0), thickness=2)

    ################################################### Straight lines (older method)
    # # 7) Hough Lines
    # lines = cv2.HoughLinesP(
    #     cropped,
    #     rho=.95,
    #     theta=np.pi / 180,
    #     threshold=20,
    #     minLineLength=15,
    #     maxLineGap=100
    # )

    # # 8) Draw averaged left/right lanes
    # line_image = np.zeros_like(frame)

    # if lines is not None:
    #     # # draw all lines (for debugging)
    #     # for line in lines:
    #     #     x1, y1, x2, y2 = line[0]

    #     #     # Avoid vertical divide-by-zero
    #     #     if x2 == x1:
    #     #         continue

    #     #     slope = (y2 - y1) / (x2 - x1)

    #     #     # Only draw if it looks like a lane (angled line)
    #     #     if abs(slope) > 0.5:
    #     #         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    #     # # Draw averaged lanes
    #     averaged_lines = average_slope_intercept(frame, lines)

    #     for x1, y1, x2, y2 in averaged_lines:
    #         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ################################################### older method end

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
