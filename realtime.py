import cv2
import numpy as np
from tensorflow.keras.models import load_model   # ✅ use load_model instead of model_from_json

# ------------------------------
# Load model directly from .h5
# ------------------------------
model = load_model("facialemotionmodel.h5")
print("✅ Model loaded successfully!")

# ------------------------------
# Haar Cascade
# ------------------------------
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ------------------------------
# Labels
# ------------------------------
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# ------------------------------
# Open webcam
# ------------------------------
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img, verbose=0)   # ✅ suppress logs
        prediction_label = labels[pred.argmax()]

        cv2.putText(im, prediction_label, 
                    (p, q-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 255), 2)

    cv2.imshow("Facial Emotion Recognition", im)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
