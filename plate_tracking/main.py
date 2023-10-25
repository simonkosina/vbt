# TODO: Cite https://www.tensorflow.org/lite/models/modify/model_maker/object_detection#run_object_detection_and_show_the_detection_results
# TODO: Can't let the plates be detected in different order. Let the user pick the right one, if multiple plates detected and based on distance, decide which is which.
# TODO: CAPTURE_SOURCE, THRESHOLD, HEIGHT, CREATE_DATAFRAME, MODEL_PATH to click args

import click
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import sys
import os

from KalmanFilter import KalmanFilter
from odt import run_odt, draw_bar_path, draw_bounding_box, calc_bounding_box_center

COLORS = [(115, 3, 252), (255, 255, 255)]
CREATE_DATAFRAME = True  # Creates a dataframe used by plot.py

# TODO: create_dataframe and dataframe path attributes
@click.command()
@click.argument('capture_source', type=str)
@click.option('--model_path', default='models/efficientdet_lite0_whole.tflite', help='Path to a TF Lite model used for object detection', type=str)
@click.option('--detection_treshold', default=0.5, help='Object detection threshold.', type=float)
@click.option('--display_image_height', default=1000, help='Displayed image height in pixels. Image width will be calculated to keep the same ratio as the original capture source.', type=int)
def main(capture_source, model_path, detection_treshold, display_image_height):
    """
    Visualize the object detection model for barbell tracking on a video
    and create a dataframe containing the detected objects their raw
    and filtered positions and velocities at specific times in the video. 
    """
    # Gather data about the video
    cap = cv2.VideoCapture(capture_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2h_ratio = float(frame_width)/float(frame_height)
    display_image_width = int(display_image_height*w2h_ratio)
    
    model_name = os.path.basename(model_path).split('.')[0]

    return
    # Initialize Kalman Filters for each tracking_id (in the app, one KF will be enough)
    kf_args = {'dt': 1/fps, 'std_acc': 1, 'xm_std': 0.015, 'ym_std': 0.015}
    kfs = {}

    # Initialize tracking variables
    frame_count = 0
    data = {'id': [], 'time': [], 'x_raw': [], 'y_raw': [],
            'x_filtered': [], 'y_filtered': [], 'dx': [], 'dy': []}

    # Store bar paths, key is object's tracking id, values are lists tuples [x, y]
    # representing the original image coordinates.
    raw_bar_paths = {}
    filtered_bar_paths = {}

    # Initialize the tflite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=16)
    interpreter.allocate_tensors()

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break

        if frame_count % 2 == 0:
            time = frame_count / fps

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = True

            results = run_odt(
                frame=img,
                interpreter=interpreter,
                threshold=detection_treshold
            )

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # TODO: Might want to keep the results
            for kf in kfs.values():
                kf.predict()

            for tracking_id, result in results.items():
                draw_bounding_box(
                    image=img,
                    tracking_id=tracking_id,
                    obj=result,
                    color=COLORS[0]
                )

                # Center in normalized image coordinates
                x_raw, y_raw = calc_bounding_box_center(
                    result['bounding_box'])

                # Initialize Kalman Filter if a new object detected
                if tracking_id not in kfs:
                    kfs[tracking_id] = KalmanFilter(
                        x_raw, y_raw, **kf_args)

                # Estimated positions and velocities
                x, y, dx, dy = kfs[tracking_id].update(
                    [[x_raw], [y_raw]]).squeeze()

                # Bounding box center in image coordinates
                raw_center = np.array(
                    [x_raw*img.shape[1], y_raw*img.shape[0]], dtype=np.int32)
                filtered_center = np.array(
                    [x*img.shape[1], y*img.shape[0]], dtype=np.int32)

                # Store and draw the bar paths
                if tracking_id in raw_bar_paths:
                    raw_bar_paths[tracking_id] = np.concatenate(
                        (raw_bar_paths[tracking_id], [raw_center]), dtype=np.int32)
                    filtered_bar_paths[tracking_id] = np.concatenate(
                        (filtered_bar_paths[tracking_id], [filtered_center]), dtype=np.int32)
                else:
                    raw_bar_paths[tracking_id] = np.array(
                        [raw_center], np.int32)
                    filtered_bar_paths[tracking_id] = np.array(
                        [filtered_center], np.int32)

                draw_bar_path(img, raw_bar_paths[tracking_id], color=COLORS[0])
                draw_bar_path(
                    img, filtered_bar_paths[tracking_id], color=COLORS[1])

                # Append data to the dataframe
                data['id'].append(tracking_id)
                data['time'].append(time)
                data['x_raw'].append(x_raw)
                data['y_raw'].append(y_raw)
                data['x_filtered'].append(x)
                data['y_filtered'].append(y)
                data['dx'].append(dx)
                data['dy'].append(dy)

            # TODO: Rep counting based on dx, dy from calman filter? Use the P matrix from K.F. to asses std in velocity/position for implementing tresholds.
            # TODO: Detect the beginning of the exercise (ask user if he'll be reracking, if yes give a signal when a new stabilised position has been achieved)

            # Show results
            img_resized = cv2.resize(img, (display_image_width, display_image_height))
            cv2.imshow("Plate Tracking", img_resized)

            # Stream Control
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # if CREATE_DATAFRAME:
    #     df_path = f'{capture_source.split(".")[0]}_idX_{model_name}.pkl'
    #     df = pd.DataFrame.from_dict(data)
    #     df.to_pickle(df_path)


if __name__ == "__main__":
    main()
