import cv2
import numpy as np

# Load image
image = cv2.imread("test_image.jpg")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Orange/brown color range
lower = np.array([5, 100, 100])
upper = np.array([20, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower, upper)

# Find contours
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 300:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Nut Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

# Show result
cv2.imshow("Nut Detection", image)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()