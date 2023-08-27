# TODO: Display the height of the wrist throughout time (Y axis)
import os
import cv2
import mediapipe as mp
import csv
import subprocess

from threading import Thread
from collections import deque
from enum import Enum

VISIBILITY_TRESHOLD = 0.9
CAPTURE_SOURCE = "data/pose/staged/2_squat_front-angled.mp4"
IM_HEIGHT_PX = 972
NUM_DELTAS = 6
FILE_OUT = 'heights.csv'
FIELD_NAMES = ['time', 'height']

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class Direction(Enum):
    UP = 0
    DOWN = 1
    STILL = 2


class ImageWriter(object):
    def __init__(
            self,
            image,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            textColor=(255, 255, 255),
            backgroundColor=(115, 3, 252),
            thickness=2,
            lineType=cv2.LINE_AA,
            horizontalPadding=10,
            verticalPadding=10
            ):
        self.image = image
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.textColor = textColor
        self.backgroundColor = backgroundColor
        self.thickness = thickness
        self.lineType = lineType
        self.horizontalPadding = horizontalPadding
        self.verticalPadding = verticalPadding

        self.n_lines = 0

    def putText(self, text):
        (text_width, text_height), _ = cv2.getTextSize(
            text=text,
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            thickness=self.thickness
        )

        y_start = self.n_lines*(text_height+self.horizontalPadding)
        y_end = (self.n_lines+1)*(text_height+self.horizontalPadding)

        cv2.rectangle(
            img=self.image,
            pt1=(0, y_start),
            pt2=(self.verticalPadding + text_width, y_end),
            color=self.backgroundColor,
            thickness=-1
        )

        cv2.putText(
            img=self.image,
            text=text,
            org=(int(self.verticalPadding/2),
                 y_end + self.fontScale - 1 - int(self.horizontalPadding/2)),
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            color=self.textColor,
            thickness=self.thickness,
            lineType=self.lineType
        )

        self.n_lines += 1


# FIXME: Needs smoothing out
# def set_direction(deltas):
#     deltas_sum = sum(deltas)

#     if deltas_sum > 0:
#         return Direction.UP
#     else:
#         return Direction.DOWN


def get_distance(landmark1, landmark2):
    x2 = (landmark1.x - landmark2.x)**2
    y2 = (landmark1.y - landmark2.y)**2
    z2 = (landmark1.z - landmark2.z)**2

    return (x2 + y2 + z2)**(1/2)


def print_heights(times: deque, heights: deque):
    rows = []

    while len(times):
        rows.append([times.popleft(), heights.popleft()])

    with open(FILE_OUT, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)


if __name__ == "__main__":
    # Initialize the CSV output file
    with open(FILE_OUT, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(FIELD_NAMES)

    thread = Thread(target=lambda: subprocess.run(["python", "plot.py"]))
    thread.start()

    # Capture data
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)
    heights = deque()
    times = deque()

    # Loop initialization
    wrist = None
    visibility = None
    distance = 0
    count = 0
    prev_results = None
    # y_deltas = [0] * NUM_DELTAS
    # direction = None

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.95
    ) as pose:
        while (cap.isOpened()):
            ret, frame = cap.read()
            update_data = False

            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    update_data = prev_results is not None
                    update_data = update_data and prev_results.pose_landmarks
                    update_data = update_data and prev_results.pose_world_landmarks.landmark[wrist].visibility > VISIBILITY_TRESHOLD
                    update_data = update_data and results.pose_world_landmarks.landmark[wrist].visibility > VISIBILITY_TRESHOLD

                    # TODO: Set only once? Or make sure that we are not computing the distance while switching wrists.
                    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z:
                        wrist = mp_pose.PoseLandmark.LEFT_WRIST
                    else:
                        wrist = mp_pose.PoseLandmark.RIGHT_WRIST

                    if update_data:
                        # TODO: Only if the difference is large enough
                        distance += get_distance(
                            prev_results.pose_world_landmarks.landmark[wrist],
                            results.pose_world_landmarks.landmark[wrist]
                        )

                        # FIXME: Smooth out the directions
                        # y_deltas[count % NUM_DELTAS] = prev_results.pose_world_landmarks.landmark[wrist].y - results.pose_world_landmarks.landmark[wrist].y
                        # direction = set_direction(y_deltas)

                    heights.append(results.pose_world_landmarks.landmark[wrist].y)
                    times.append(round(count/fps, 2))

                    visibility = results.pose_world_landmarks.landmark[wrist].visibility

                im_resized = cv2.resize(image, (IM_WIDTH_PX, IM_HEIGHT_PX))

                writer = ImageWriter(im_resized)
                writer.putText(f"wrist: {'left' if wrist == mp_pose.PoseLandmark.LEFT_WRIST else 'right'}")
                writer.putText(f"distance: {distance:.2f}")
                writer.putText(f"time: {(count/fps):.2f}")
                writer.putText(f"visibility: {visibility if visibility else '-'}")
                # writer.putText(f"direction: {direction if direction else '-'}")

                cv2.imshow(str(CAPTURE_SOURCE), im_resized)

                if count % 5 == 0:
                    print_heights(times, heights)

                # Stream Control
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                prev_results = results
                count += 1
            else:
                break

    print_heights(times, heights)

    cap.release()
    cv2.destroyAllWindows()

    thread.join()
