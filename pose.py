import cv2
import mediapipe as mp

VISIBILITY_TRESHOLD = 0.9
CAPTURE_SOURCE = 0

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class ImageWriter(object):
    def __init__(
            self,
            image,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            textColor=(255, 255, 255),
            backgroundColor=(115, 3, 252),
            thickness=2,
            lineType=cv2.LINE_AA,
            ):
        self.image = image
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.textColor = textColor
        self.backgroundColor = backgroundColor
        self.thickness = thickness
        self.lineType = lineType

        self.n_lines = 0 

    def putText(self, text):
        (text_width, text_height), _ = cv2.getTextSize(
            text=text,
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            thickness=self.thickness
        )

        # TODO: padding
        y_start = int(self.n_lines*text_height + self.fontScale - 1)
        y_end = int((self.n_lines+1)*text_height + self.fontScale - 1)

        cv2.rectangle(
            img=self.image,
            pt1=(0, y_start),
            pt2=(10 + text_width, y_end),
            color=self.backgroundColor,
            thickness=-1
        )

        cv2.putText(
            img=self.image,
            text=text,
            org=(0, y_end),
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            color=self.textColor,
            thickness=self.thickness,
            lineType=self.lineType
        )

        self.n_lines += 1


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

    wrist = None
    distance = 0
    time = 0
    prev_results = None

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.95
    ) as pose:
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z:
                        wrist = mp_pose.PoseLandmark.LEFT_WRIST
                    else:
                        wrist = mp_pose.PoseLandmark.RIGHT_WRIST

                    if prev_results and prev_results.pose_landmarks and results.pose_world_landmarks.landmark[wrist].visibility > VISIBILITY_TRESHOLD:
                        # TODO: Only if the difference is large enough
                        distance += get_distance(
                            prev_results.pose_world_landmarks.landmark[wrist],
                            results.pose_world_landmarks.landmark[wrist]
                        )

                    visibility = results.pose_world_landmarks.landmark[wrist].visibility

                writer = ImageWriter(image)
                writer.putText("Bim Bam")
                # writer.putText("ahoj ako sa mas")

                cv2.imshow(str(CAPTURE_SOURCE), image)

                # Stream Control
                key = cv2.waitKey(1)
                if key & 0xFF == ord('m'):
                    print(frame)
                if key & 0xFF == ord('q'):
                    break

                prev_results = results
                frame += 1
                time += 1 / fps
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
