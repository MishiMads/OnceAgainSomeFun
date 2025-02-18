import os
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Perform object detection
results = model("image2.png")

# Define confidence threshold
CONF_THRESHOLD = 0.5  # Only save objects with confidence > 50%

# Create a folder to store cropped images
output_folder = "detected_people"
os.makedirs(output_folder, exist_ok=True)

# Load original image
image = cv2.imread("image2.png")

# Counter for naming cropped images
person_count = 0

# Process each detected object
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence score
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

        # Check if detected object is a person (class ID = 0)
        if cls == 0 and conf > CONF_THRESHOLD:
            print(f"Saving person {person_count + 1} with confidence: {conf:.2f}")

            # Crop the person from the image
            cropped_person = image[y1:y2, x1:x2]

            # Save cropped image
            person_filename = os.path.join(output_folder, f"person_{person_count}.jpg")
            cv2.imwrite(person_filename, cropped_person)

            person_count += 1

print(f"Saved {person_count} people in '{output_folder}' folder.")
