import cv2
import httpx
from datetime import datetime
import argparse
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
argparser.add_argument(
    "--headless",
    action="store_true",
    dest="headless",
    help="Runs face detector in headless mode",
)
args = argparser.parse_args()


BASE_URL = "http://127.0.0.1:8000"
HAS_CONNECTION = True

try:
    EXPERIMENT_ID = httpx.get(f"{BASE_URL}/experiments/last/").json()["id"]
    logging.info(f"Connected to {BASE_URL}")
except httpx.ConnectError:
    logging.info("Couldn't connect to remote host, running in offline mode")
    HAS_CONNECTION = False


def post_event(detected):
    if not HAS_CONNECTION:
        return

    status = "I see you" if detected else "I don't see you"
    event = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "name": "CAMERA",
        "experiment": EXPERIMENT_ID,
    }
    httpx.post(f"{BASE_URL}/events/", data=event)
    logging.debug(
        f"POST event with status and experiment id: {status} - {EXPERIMENT_ID}"
    )


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Couldn't open camera, check if it's connected")


face_cascade = cv2.CascadeClassifier(
    "./haarcascade_models/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("./haarcascade_models/haarcascade_eye.xml")

face_in_last_iteration = False
skiped_frames = 0
while True:
    # Capture frames
    _, img = cap.read()
    skiped_frames += 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.08, 7, minSize=(100, 100))

    # Skiped some frames to reduce the amount of possible events sent
    if skiped_frames == 4:
        skiped_frames = 0
        if len(faces) != 0 and not face_in_last_iteration:
            post_event(detected=True)
            face_in_last_iteration = True
        elif len(faces) == 0 and face_in_last_iteration:
            post_event(detected=False)
            face_in_last_iteration = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.08, 7, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if not args.headless:
        cv2.imshow("img", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
