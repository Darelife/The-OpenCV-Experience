import os
import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def getCammeraInput() -> cv2.VideoCapture:
    return cv2.VideoCapture(0)


def getFrame(cam: cv2.VideoCapture):
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        exit()
    return frame


def showFrame(frame):
    cv2.imshow("frame", frame)


def releaseCammera(cam: cv2.VideoCapture):
    cam.release()
    cv2.destroyAllWindows()


def get_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume


def fingers_open(hand_landmarks):
    # Check if fingers are open
    tips_ids = [4, 8, 12, 16, 20]
    open_fingers = 0
    for tip_id in tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            open_fingers += 1
    return open_fingers == 5


def main():
    cam = getCammeraInput()
    hands = mp_hands.Hands()
    volume = get_volume_control()
    while True:
        frame = getFrame(cam)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                if fingers_open(hand_landmarks):
                    volume.SetMasterVolumeLevelScalar(
                        min(volume.GetMasterVolumeLevelScalar() + 0.009, 1.0), None
                    )
                else:
                    volume.SetMasterVolumeLevelScalar(
                        max(volume.GetMasterVolumeLevelScalar() - 0.009, 0.0), None
                    )

        showFrame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    releaseCammera(cam)
    hands.close()


if __name__ == "__main__":
    main()
