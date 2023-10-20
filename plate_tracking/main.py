# TODO: Better handle the original/resized img coordinates in bar_paths and in graphs
# TODO: Cite https://www.tensorflow.org/lite/models/modify/model_maker/object_detection#run_object_detection_and_show_the_detection_results
# TODO: Can't let the plates be detected in different order. Let the user pick the right one, if multiple plates detected and based on distance, decide which is which.
# TODO: CAPTURE_SOURCE, THRESHOLD, HEIGHT, CREATE_DATAFRAME, MODEL_PATH to click args

import cv2
import tensorflow as tf
import pandas as pd
import os

from KalmanFilter import KalmanFilter
from odt import run_odt, draw_results, calc_bounding_box_center

TRACKING_ID = 0 # User will be able to pick in the application
MODEL_PATH = "plate_tracking/models/efficientdet_lite0_whole.tflite"
# CAPTURE_SOURCE = "plate_tracking/samples/cut/016_squat_8_reps.mp4"
CAPTURE_SOURCE = "plate_tracking/samples/raw/024_dl_4_reps.mp4"
IM_HEIGHT_PX = 1000
DETECTION_TRESHOLD = 0.5

CREATE_DATAFRAME = True # Creates a dataframe used by plot.py
DATAFRAME_FILENAME = CAPTURE_SOURCE.split('.')[0] + '.pkl'

if __name__ == "__main__":
    # Gather data about the video
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)

    # Initialize Kalman Filters for each tracking_id (in the app, one KF will be enough)
    kf_args = {'dt': 1/fps, 'ux': 0, 'uy': 0, 'std_acc': 1, 'xm_std': 0.01, 'ym_std': 0.01}
    kfs = {}

    # Initialize tracking variables
    frame_count = 0
    data = {'id': [], 'time': [], 'x_raw': [], 'y_raw': [], 'x_filtered': [], 'y_filtered': []}

    # Store bar paths, key is object's tracking id, values are lists tuples [x, y]
    # representing the original image coordinates.
    bar_paths = {}

    # Initialize the tflite interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=16)
    interpreter.allocate_tensors()

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1 # TODO: Remove frame_count conditions

        if not ret:
            break

        if frame_count % 2 == 0:
            time = frame_count / fps

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = True

            results = run_odt(
                frame=img,
                interpreter=interpreter,
                threshold=DETECTION_TRESHOLD
            )

            img = draw_results(
                image=img,
                results=results,
                bar_paths=bar_paths
            )

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_resized = cv2.resize(img, (IM_WIDTH_PX, IM_HEIGHT_PX))

            if CREATE_DATAFRAME:
                for tracking_id, result in results.items():
                    x, y = calc_bounding_box_center(result['bounding_box'])
                    # TODO: Sort out the original/resized image coordinates
                    # TODO: Plot the Kalman estimates
                    # TODO: Display the estimated bar_bath

                    if tracking_id not in kfs:
                        kfs[tracking_id] = KalmanFilter(x, y, **kf_args)
                    
                    kfs[tracking_id].predict()
                    xk, yk = kfs[tracking_id].update([[x], [y]])
                    
                    # TODO: in the return statement
                    xk = xk[0]
                    yk = yk[0]
 
                    data['id'].append(tracking_id)
                    data['time'].append(time)
                    data['x_raw'].append(x)
                    data['y_raw'].append(y)
                    data['x_filtered'].append(xk)
                    data['y_filtered'].append(yk)

            # TODO: Run the prediction step if the object isn't detected.

            # TODO: Rep counting based on dx, dy from calman filter?

            # Show results
            cv2.imshow("Plate Tracking", img_resized)

            # Stream Control
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    if CREATE_DATAFRAME:
        df = pd.DataFrame.from_dict(data)
        df.to_pickle(DATAFRAME_FILENAME)
