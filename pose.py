# TODO: Display the height of the wrist throughout time (Y axis)
import cv2
import mediapipe as mp
import numpy as np

from math import floor
from collections import deque
from enum import Enum

HEIGHT_TRESHOLD = 0.025
VISIBILITY_TRESHOLD = 0.9
# CAPTURE_SOURCE = "data/pose/staged/2_squat_front-angled.mp4"
CAPTURE_SOURCE = "data/pose/gym/cut_deadlift_6reps_20200827_150916.mp4"
# CAPTURE_SOURCE = "data/pose/gym/cut_rdl_9reps_20230822_064315542.mp4"
IM_HEIGHT_PX = 800
NUM_DELTAS = 6

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp.solutions.pose.Pose()


class Phase(Enum):
    CONCENTRIC = 0
    ECCENTRIC = 1
    HOLD = 2


STARTING_PHASE = Phase.CONCENTRIC


class RepCounter(object):
    @staticmethod
    def calculate_min_treshold(height_min):
        return height_min + HEIGHT_TRESHOLD

    @staticmethod
    def calculate_max_treshold(height_max):
        return height_max - HEIGHT_TRESHOLD

    @staticmethod
    def opposite_phase(phase):
        if phase == Phase.CONCENTRIC:
            return Phase.ECCENTRIC
        elif phase == Phase.ECCENTRIC:
            return Phase.CONCENTRIC

    def __init__(self, starting_phase):
        if starting_phase not in [Phase.CONCENTRIC, Phase.ECCENTRIC]:
            raise ValueError('Starting phase must be either concentri or eccentric')

        self.starting_phase = starting_phase
        self.curr_phase = Phase.HOLD
        self.prev_phase = self.opposite_phase(starting_phase)

        self.first_rep = True
        self.height = None
        self.min_treshold = np.inf
        self.max_treshold = -np.inf

        self.num_holds = 0

        self.height_min = np.inf
        self.height_max = -np.inf

    @property
    def num_reps(self):
        return floor(self.num_holds/2)

    def _first_concentric_rep(self, height):
        if self.curr_phase == Phase.HOLD and height < self.max_treshold:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.prev_phase = Phase.HOLD

        if self.curr_phase != Phase.HOLD and height > self.min_treshold:
            self.curr_phase = self.opposite_phase(self.curr_phase)
            self.prev_phase = Phase.HOLD
            self.first_rep = False
            self.num_holds += 1

    def _first_eccentric_rep(self, height):
        if self.curr_phase == Phase.HOLD and height > self.min_treshold:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.prev_phase = Phase.HOLD

        if self.curr_phase != Phase.HOLD and height < self.max_treshold:
            self.curr_phase = self.opposite_phase(self.curr_phase)
            self.prev_phase = Phase.HOLD
            self.first_rep = False
            self.num_holds += 1

    def update_min_treshold(self, height):
        if self.height_min > height:
            self.height_min = height

        new_treshold = self.calculate_min_treshold(height)

        if self.min_treshold > new_treshold:
            self.min_treshold = new_treshold

    def update_max_treshold(self, height):
        if self.height_max < height:
            self.height_max = height

        new_treshold = self.calculate_max_treshold(height)

        if self.max_treshold < new_treshold:
            self.max_treshold = new_treshold

    def update(self, height):
        self.update_min_treshold(height)
        self.update_max_treshold(height)

        # Handles first rep
        if self.first_rep and self.starting_phase == Phase.CONCENTRIC:
            self._first_concentric_rep(height)
            return

        if self.first_rep and self.starting_phase == Phase.ECCENTRIC:
            self._first_eccentric_rep(height)
            return

        # Handles subsequent reps
        if height > self.min_treshold and height < self.max_treshold and self.curr_phase == Phase.HOLD:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.prev_phase = Phase.HOLD

        if (height < self.min_treshold or height > self.max_treshold) and self.curr_phase != Phase.HOLD:
            self.prev_phase = self.curr_phase
            self.curr_phase = Phase.HOLD
            self.num_holds += 1


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


def calculate_distance(landmark1, landmark2):
    x2 = (landmark1.x - landmark2.x)**2
    y2 = (landmark1.y - landmark2.y)**2
    z2 = (landmark1.z - landmark2.z)**2

    return (x2 + y2 + z2)**(1/2)


def height_plot(x_list, y_list, y_min, y_max, image_height=150, vpadding_px=10):
    y_size = image_height - vpadding_px
    im_graph = np.zeros((image_height, IM_WIDTH_PX, 3), dtype=np.uint8)

    if len(y_list) < 2:
        return im_graph

    xs = np.array(x_list[-IM_WIDTH_PX:], dtype=np.int32)
    ys = np.array(y_list[-IM_WIDTH_PX:], dtype=np.float32)

    # Adjust x coordinate when data starts to overflow
    if len(x_list) - IM_WIDTH_PX > 0:
        xs = xs - xs[0]

    def adjust(arr):
        """Normalize the array into the [vpadding_px/2, y_size + vpadding_px/2] interval"""
        return (arr - y_min)/(y_max - y_min)*y_size + vpadding_px/2

    ys_scaled = adjust(ys)
    min_scaled = adjust(RepCounter.calculate_min_treshold(y_min))
    max_scaled = adjust(RepCounter.calculate_max_treshold(y_max))

    # Plot height
    curve = np.column_stack((xs, ys_scaled.astype(np.int32)))
    cv2.polylines(im_graph, [curve], False, (0, 255, 255))

    # Plot tresholds
    min_treshold = np.array([[0, min_scaled], [IM_WIDTH_PX, min_scaled]], dtype=np.int32)
    max_treshold = np.array([[0, max_scaled], [IM_WIDTH_PX, max_scaled]], dtype=np.int32)

    cv2.polylines(im_graph, [min_treshold], False, (0, 255, 0))  # Green
    cv2.polylines(im_graph, [max_treshold], False, (0, 0, 255))  # Red

    return im_graph


if __name__ == "__main__":
    # Capture data
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)
    heights = []
    counts = []
    y_min = np.inf
    y_max = -np.inf

    # Loop initialization
    wrist = None
    visibility = None
    distance = 0
    count = 0
    prev_results = None
    rep_counter = RepCounter(STARTING_PHASE)

    with mp_pose.Pose(
        model_complexity=1,
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
                    update_data = update_data and prev_results.pose_world_landmarks
                    update_data = update_data and prev_results.pose_world_landmarks.landmark[wrist].visibility > VISIBILITY_TRESHOLD
                    update_data = update_data and results.pose_world_landmarks.landmark[wrist].visibility > VISIBILITY_TRESHOLD

                    # TODO: Set only once? Or make sure that we are not computing the distance while switching wrists.
                    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z:
                        wrist = mp_pose.PoseLandmark.LEFT_WRIST
                    else:
                        wrist = mp_pose.PoseLandmark.RIGHT_WRIST

                    if update_data:
                        # TODO: Only if the difference is large enough?
                        distance += calculate_distance(
                            prev_results.pose_world_landmarks.landmark[wrist],
                            results.pose_world_landmarks.landmark[wrist]
                        )

                    # FIXME: See what's happening with the world coordinates? They seem to fluctuate too much compared to the image coordinates.
                    # y = results.pose_world_landmarks.landmark[wrist].y
                    y = results.pose_landmarks.landmark[wrist].y
                    rep_counter.update(height=y)

                    heights.append(y)
                    counts.append(count)

                    visibility = results.pose_world_landmarks.landmark[wrist].visibility

                im_resized = cv2.resize(image, (IM_WIDTH_PX, IM_HEIGHT_PX))

                writer = ImageWriter(im_resized)
                writer.putText(f"wrist: {'left' if wrist == mp_pose.PoseLandmark.LEFT_WRIST else 'right'}")
                writer.putText(f"visibility: {visibility if visibility else '-'}")
                writer.putText(f"phase: {rep_counter.curr_phase}")
                writer.putText(f"reps: {rep_counter.num_reps}")

                im_graph = height_plot(counts, heights, rep_counter.height_min, rep_counter.height_max)
                im_vconcat = cv2.vconcat([im_resized, im_graph])
                cv2.imshow(str(CAPTURE_SOURCE), im_vconcat)

                # Stream Control
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                prev_results = results
                count += 1
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
