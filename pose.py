# TODO: Display the height of the wrist throughout time (Y axis)
import cv2
import mediapipe as mp
import numpy as np

from RepCounter import RepCounter, Phase
from ImageWriter import ImageWriter
from helpers import calculate_distance, height_plot

VISIBILITY_TRESHOLD = 0.9
# CAPTURE_SOURCE = "data/pose/staged/2_bp_front-angled.mp4"
# CAPTURE_SOURCE = "data/pose/staged/2_bp_back-angled.mp4"
# CAPTURE_SOURCE = "data/pose/staged/1_squat_back-angled.mp4"
CAPTURE_SOURCE = "data/pose/gym/cut_deadlift_6reps_20200827_150916.mp4"
# CAPTURE_SOURCE = "data/pose/gym/cut_deadlift_8reps_20230203_130125.mp4"
# CAPTURE_SOURCE = "data/pose/gym/cut_rdl_9reps_20230822_064315542.mp4"
IM_HEIGHT_PX = 800
NUM_DELTAS = 6
STARTING_PHASE = Phase.CONCENTRIC

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


if __name__ == "__main__":
    # Capture data
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)

    # Loop initialization
    distances = []  # FIXME: Might be needing too much memory. Needed for the first rep, can just accumulate without remembering for subsequent reps.
    heights = []
    counts = []
    y_min = np.inf
    y_max = -np.inf
    wrist = None
    visibility = None
    distance = 0
    lr_distance = 0
    lr_time = 0
    frame_count = 0
    prev_results = None
    rep_counter = RepCounter(STARTING_PHASE)
    concentric_start_frame = 0
    concentric_end_frame = 0
    acv = None
    first_rep_processing = False

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

                    # Remember if the first rep ended to alter statistics computations.
                    first_rep_processing = rep_counter.first_rep 

                    # FIXME: See what's happening with the world coordinates? They seem to fluctuate too much compared to the image coordinates.
                    # y = results.pose_world_landmarks.landmark[wrist].y
                    y = results.pose_landmarks.landmark[wrist].y
                    rep_counter.update(height=y)

                    if update_data:
                        # TODO: Only if the difference is large enough?
                        # FIXME: Only count the distance if it's actually needed.
                        distances.append(calculate_distance(
                            prev_results.pose_world_landmarks.landmark[wrist],
                            results.pose_world_landmarks.landmark[wrist]
                        ))

                        if rep_counter.concentric_start:
                            concentric_start_frame = frame_count

                        if rep_counter.concentric_end:
                            concentric_end_frame = frame_count

                            # Discard the "hold" part from the first rep
                            if first_rep_processing:
                                # Find the frame where the first rep crossed the min/max treshold
                                if STARTING_PHASE == Phase.CONCENTRIC:
                                    def evaluate(x):
                                        return x < rep_counter.min_treshold
                                else:
                                    def evaluate(x):
                                        return x > rep_counter.max_treshold

                                for index, height in enumerate(heights):
                                    if evaluate(height):
                                        break

                                lr_distance = sum(distances[concentric_start_frame:index+1])
                                lr_time = (index - concentric_start_frame) / fps
                            else:
                                lr_distance = sum(distances[concentric_start_frame:concentric_end_frame+1])
                                lr_time = (concentric_end_frame - concentric_start_frame) / fps

                            acv = lr_distance / lr_time  # m/s

                    # Data for the height plot
                    heights.append(y)
                    counts.append(frame_count)

                    visibility = results.pose_world_landmarks.landmark[wrist].visibility

                im_resized = cv2.resize(image, (IM_WIDTH_PX, IM_HEIGHT_PX))

                writer = ImageWriter(im_resized)
                writer.putText(f"wrist: {'left' if wrist == mp_pose.PoseLandmark.LEFT_WRIST else 'right'}")
                writer.putText(f"visibility: {visibility if visibility else '-'}")
                writer.putText(f"phase: {rep_counter.curr_phase}")
                writer.putText(f"reps: {rep_counter.num_reps}")
                writer.putText(f"lr time: {lr_time}")
                writer.putText(f"lr distance: {lr_distance}")
                writer.putText(f"lr ACV: {acv if acv else '-'}")

                im_graph = height_plot(counts, heights, rep_counter.height_min, rep_counter.height_max, IM_WIDTH_PX)
                im_vconcat = cv2.vconcat([im_resized, im_graph])
                cv2.imshow(str(CAPTURE_SOURCE), im_vconcat)

                # Stream Control
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                if not update_data:
                    distances.append(0)

                prev_results = results
                frame_count += 1
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
