import numpy as np
import cv2
import sys

VIDEO = "./teste5.mp4"

algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
a = 3
algorithm_type = algorithm_types[a]

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation

def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Detector inválido')
    sys.exit(1)

# -------------------------------------------------------------------------------------------------------------------------

linha_ROI = 150  # Position of the counting line
latas = 0

# Variables for tracking
detected_cans = []
memory = {}  # Memory for tracked objects (to avoid double counting)
max_memory_frames = 10  # The maximum number of frames to remember a detected can
min_dist = 50  # Minimum distance between consecutive detections of the same can

# Buffer for consecutive detection validation
buffer = []  # Buffer to store detections over frames
buffer_size = 2  # Number of consecutive frames to validate detections

def is_new_detection(cx, cy):
    """
    This function checks if the detected centroid (cx, cy) is a new object,
    based on the proximity to already tracked objects in memory.
    """
    for (mcx, mcy), frames in list(memory.items()):
        dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
        if dist < min_dist:
            return False
    return True

def update_memory():
    """
    Update memory by removing cans that have been tracked for too many frames.
    """
    for key in list(memory.keys()):
        memory[key] -= 1
        if memory[key] <= 0:
            del memory[key]

def validate_detection(cx, cy):
    """
    Check if a can has been detected consistently across multiple frames
    by storing detections in a buffer and validating them.
    """
    buffer.append((cx, cy))
    if len(buffer) > buffer_size:
        buffer.pop(0)  # Remove the oldest detection from the buffer

    # Check if the same can (same coordinates) appears in enough consecutive frames
    occurrences = sum(1 for (bx, by) in buffer if np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) < min_dist)
    return occurrences >= (buffer_size - 2)  # Allow slight variations

def set_info(detected_cans):
    global latas
    for (cx, cy) in detected_cans:
        if validate_detection(cx, cy) and is_new_detection(cx, cy) and (linha_ROI - 10) < cy < (linha_ROI + 10):
            latas += 1
            memory[(cx, cy)] = max_memory_frames  # Add new detection to memory
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            print(f"Latas detectadas até o momento: {latas}")

def show_info(frame, mask):
    text = f'Latas: {latas}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    # cv2.imshow("Detectar", mask)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO)

# Define the background subtractor
background_subtractor = Subtractor(algorithm_type)

while True:
    ok, frame = cap.read()  # Read each frame from the video
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use background subtraction to get the foreground mask
    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')

    # Detect circles in the blurred image using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.25, minDist=70, param1=70, param2=20, minRadius=37, maxRadius=41)

    detected_cans = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Draw the circle and its centroid
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            detected_cans.append((x, y))

    # Draw the counting line
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)

    set_info(detected_cans)
    show_info(frame, mask)

    update_memory()  # Remove old entries from memory

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
