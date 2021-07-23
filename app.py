import cv2
import httpx
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
HAS_CONNECTION = True

try:
    EXPERIMENT_ID = httpx.get(f"{BASE_URL}/experiments/last").json()["id"]
except httpx.ConnectError:
    HAS_CONNECTION = False


face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Couldn't open camera, check if it's connected")


async def post_event(detected):
    event = {
        "timestamp": datetime.now().isoformat(),
        "status": "I see you" if detected else "I don't see you",
        "name": "CAMERA",
        "experiment": EXPERIMENT_ID,
    }
    async with httpx.AsyncClient() as client:
        client.post(f"{BASE_URL}/events/", data=event)


post_event(detected=True)
face_in_last_iteration = False

while True:
    # Capture frames
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces and not face_in_last_iteration:
        post_event(detected=True)
        face_in_last_iteration = True
    elif not faces and face_in_last_iteration:
        post_event(detected=False)
        face_in_last_iteration = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("img", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
