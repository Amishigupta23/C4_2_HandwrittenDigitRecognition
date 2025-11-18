import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("digit_model.h5")

# Initialize video
cap = cv2.VideoCapture(0)

# Prediction smoothing queue
pred_queue = deque(maxlen=5)

# Fixed ROI coordinates
x1, y1, x2, y2 = 150, 100, 350, 300  # adjust for your camera setup

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw fixed green rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to isolate digit
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Resize and center digit
    resized = cv2.resize(thresh, (28, 28))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(input_data)
    digit = np.argmax(prediction)

    # Smooth prediction
    pred_queue.append(digit)
    smooth_digit = max(set(pred_queue), key=pred_queue.count)

    # Display
    cv2.putText(frame, f"Predicted: {smooth_digit}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
