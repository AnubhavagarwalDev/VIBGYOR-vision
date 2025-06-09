import cv2
import numpy as np

# Define the HSV color ranges for each VIBGYOR color
color_ranges = {
    'violet': {'lower': np.array([130, 50, 50]), 'upper': np.array([160, 255, 255])},
    'indigo': {'lower': np.array([110, 50, 50]), 'upper': np.array([130, 255, 255])},
    'blue': {'lower': np.array([90, 50, 50]), 'upper': np.array([110, 255, 255])},
    'green': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
    'yellow': {'lower': np.array([20, 50, 50]), 'upper': np.array([35, 255, 255])},
    'orange': {'lower': np.array([10, 50, 50]), 'upper': np.array([20, 255, 255])},
    'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
            'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])}  # Combined red ranges
}

# Define colors for drawing rectangles (BGR format)
color_bgr = {
    'violet': (255, 0, 255),
    'indigo': (75, 0, 130),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'red': (0, 0, 255)
}

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)

    # Create a copy of the original frame to display the contours separately
    contours_frame = frame.copy()

    # Convert the frame to HSV for color detection
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through the color ranges to detect each color in VIBGYOR
    for color_name, limits in color_ranges.items():
        # Handle special case for red (combining two ranges)
        if color_name == 'red':
            mask1 = cv2.inRange(hsvImage, limits['lower1'], limits['upper1'])
            mask2 = cv2.inRange(hsvImage, limits['lower2'], limits['upper2'])
            mask = mask1 | mask2
        else:
            lowerLimit = limits['lower']
            upperLimit = limits['upper']
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        # Detect the contours for each color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set a minimum area to avoid small noise
        min_area = 500
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Draw bounding rectangles with different colors for VIBGYOR on the contours frame
                cv2.rectangle(contours_frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                # Add text label for the color
                cv2.putText(contours_frame, color_name.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr[color_name], 2)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Display the frame with detected contours
    cv2.imshow('Contours Frame', contours_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
