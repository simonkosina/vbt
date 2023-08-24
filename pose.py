import cv2
import mediapipe as mp

from enum import Enum

VISIBILITY_TRESHOLD = 0.9
CAPTURE_SOURCE = "data/staged/1_squat_front-angled.mp4"
IM_HEIGHT_PX = 972
NUM_DELTAS = 5

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class RepPhase(Enum):
    CONCENTRIC = 0
    ECCENTRIC = 1
    ISOMETRIC = 2

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


# TODO: Maybe implement a delta treshold?
# FIXME: The directions don't seem to be accurate enough
def set_direction(
        deltas,
        treshold=0.5
    ):
    negative, positive = 0, 0

    for delta in deltas:
        if not delta:
            continue

        if delta < 0:
            negative += 1
        elif delta > 0:
            positive += 1

    percentage_negative = 0 if negative == 0 else len(deltas) / negative
    percentage_positive = 0 if positive == 0 else len(deltas) / positive

    if percentage_negative > treshold:
        return Direction.DOWN
    elif percentage_positive > treshold:
        return Direction.UP
    else:
        return Direction.STILL


def get_distance(landmark1, landmark2):
    x2 = (landmark1.x - landmark2.x)**2
    y2 = (landmark1.y - landmark2.y)**2
    z2 = (landmark1.z - landmark2.z)**2

    return (x2 + y2 + z2)**(1/2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(width)/float(height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)

    phase = RepPhase.CONCENTRIC  # TODO: let the user pick the starting phase
    wrist = None
    visibility = None
    distance = 0
    time = 0
    prev_results = None
    y_deltas = [None] * NUM_DELTAS
    direction = None

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

                        y_deltas[time % NUM_DELTAS] = prev_results.pose_world_landmarks.landmark[wrist].y - results.pose_world_landmarks.landmark[wrist].y
                        print(y_deltas)
                        direction = set_direction(y_deltas)

                    visibility = results.pose_world_landmarks.landmark[wrist].visibility

                im_resized = cv2.resize(image, (IM_WIDTH_PX, IM_HEIGHT_PX))

                writer = ImageWriter(im_resized)
                writer.putText(f"wrist: {'left' if wrist == mp_pose.PoseLandmark.LEFT_WRIST else 'right'}")
                writer.putText(f"distance: {distance:.2f}")
                writer.putText(f"time: {(time/fps):.2f}")
                writer.putText(f"visibility: {visibility if visibility else '-'}")
                writer.putText(f"direction: {direction if direction else '-'}")

                cv2.imshow(str(CAPTURE_SOURCE), im_resized)

                # Stream Control
                key = cv2.waitKey(1)
                if key & 0xFF == ord('m'):
                    print(frame)
                if key & 0xFF == ord('q'):
                    break

                prev_results = results
                time += 1
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
