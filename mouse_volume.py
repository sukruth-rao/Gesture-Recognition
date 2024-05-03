import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



def gesture_recognition():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.GetMute() # type: ignore
    volume.GetMasterVolumeLevel() # type: ignore
    volumeRange = volume.GetVolumeRange() # type: ignore
    vol = 0
    volumeBar = 400
    volumePercent = 0
    muteStatus = False

    mp_drawing = mp.solutions.drawing_utils # type: ignore
    mp_hands = mp.solutions.hands # type: ignore

    previousTime = 0
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            lml = []
            xl = []
            yl = []
            box = []

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Step 2: Create lists of coordinates from extracted landmarks
                for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                    h, w, _ = image.shape
                    xc, yc = int(lm.x * w), int(lm.y * h)
                    lml.append([id, xc, yc])
                    xl.append(xc)
                    yl.append(yc)

                #  Step 3: Obtain coordinates thumb and index finger tips and draw circles on the and a line between them
                x1, y1 = lml[4][1], lml[4][2]
                x2, y2 = lml[8][1], lml[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(image, (x1, y1), 10, (255, 0, 128), cv2.FILLED)
                cv2.circle(image, (x2, y2), 10, (255, 0, 128), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 128), 3)
                # cv2.circle(image, (cx, cy), 10, (255, 0, 128), cv2.FILLED)
                distance = math.hypot(x2 - x1, y2 - y1)
                # cv2.putText(image, str(int(distance)), (cx+30, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 128), 3)

                # Step 4: Create an activation function to check the hand size
                xmin, xmax = min(xl), max(xl)
                ymin, ymax = min(yl), max(yl)
                box = xmin, ymin, xmax, ymax
                cv2.rectangle(image, (box[0] - 20, box[1] - 20),
                                (box[2] + 20, box[3] + 20), (255, 255, 0), 2)
                area = (box[2] - box[0]) * (box[3] - box[1]) // 100

                if 200 < area < 1000:
                    cv2.putText(image, 'GestureControl On', (0, 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, str(
                        int(area)), (box[1] + 50, box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Step 5: Compute volume and draw volume information
                    volumeBar = int(np.interp(distance, [50, 200], [400, 150]))
                    volumePercent = int(np.interp(distance, [50, 200], [0, 100]))

                    cv2.rectangle(image, (w - 50, 150),
                                    (w - 80, 400), (255, 255, 255), 2)
                    if 21 < volumePercent < 50:
                        cv2.rectangle(image, (w - 50, int(volumeBar)),
                                        (w - 80, 400), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f'{int(volumePercent)} %', (w - 100, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 255, 0), 2)
                    elif 51 < volumePercent < 80:
                        cv2.rectangle(image, (w - 50, int(volumeBar)),
                                    (w - 80, 400), (0, 255, 255), cv2.FILLED)
                        cv2.putText(image, f'{int(volumePercent)} %', (w - 100, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 255, 255), 2)
                    elif volumePercent > 81:
                        cv2.rectangle(image, (w - 50, int(volumeBar)),
                                    (w - 80, 400), (0, 0, 255), cv2.FILLED)
                        cv2.putText(image, f'{int(volumePercent)} %', (w - 100, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 2)
                    elif volumePercent < 20:
                        cv2.rectangle(image, (w - 50, int(volumeBar)),
                                    (w - 80, 400), (255, 255, 0), cv2.FILLED)
                        cv2.putText(image, f'{int(volumePercent)} %', (w - 100, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (255, 255, 0), 2)

                    cVol = int(volume.GetMasterVolumeLevelScalar() * 100) # type: ignore
                    cv2.putText(image, f'Current Volume: {int(cVol)}', (0, 60), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 255, 255), 2)

                    # Step 6: Create Finger Check Function
                    fCount = []
                    for fid in range(8, 21, 4):
                        if lml[fid][2] < lml[fid - 2][2]:
                            fCount.append(1)
                        else:
                            fCount.append(0)

                    # Step 7: Create Set Volume and Mute/ Unmute Function
                    if fCount[3] == 0 and fCount[2] == 1 and fCount[1] == 1 and fCount[0] == 1:
                        volume.SetMasterVolumeLevelScalar(volumePercent / 100, None) # type: ignore
                        cv2.putText(image, 'Volume Set', (0, 90),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        colorVol = (0, 255, 0)
                    elif fCount[3] == 1 and fCount[2] == 0 and fCount[1] == 0 and muteStatus == False:
                        volume.SetMute(1, None) # type: ignore
                        cv2.putText(image, 'Muted', (0, 90),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        muteStatus = True
                    elif fCount[3] == 0 and fCount[2] == 0 and fCount[1] == 0 and muteStatus == True:
                        volume.SetMute(0, None) # type: ignore
                        cv2.putText(image, 'Unmuted', (0, 90),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        muteStatus = False

                    if muteStatus == True:
                        cv2.putText(image, "Muted", (0, 120),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                else:
                    cv2.putText(image, 'GestureControl Off', (0, 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, str(
                        int(area)), (box[1] + 50, box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                # Optional Step: FPS Counter
                currentTime = time.time()
                fps = 1 / (currentTime - previousTime)
                previousTime = currentTime
                cv2.putText(image, f'FPS: {int(fps)}', (w-150, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 255, 255), 2)

                cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


gesture_recognition()