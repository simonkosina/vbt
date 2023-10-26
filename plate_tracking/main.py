# TODO: Cite https://www.tensorflow.org/lite/models/modify/model_maker/object_detection#run_object_detection_and_show_the_detection_results

import click
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import os

from MovingAverage import MovingAverage
from KalmanFilter import KalmanFilter
from odt import run_odt, draw_bar_path, draw_bounding_box, calc_bounding_box_center, calc_normalized_diameter

COLORS = [(115, 3, 252), (255, 255, 255)]


@click.command()
@click.argument('src', type=str, nargs=-1)
@click.option('--model', default='models/efficientdet_lite0_whole.tflite', help='Path to a TF Lite model used for object detection.', type=str)
@click.option('--diameter', default=0.45, help='Diameter of the weight plate used in meters.', type=float)
@click.option('--detection_treshold', default=0.5, help='Object detection threshold.', type=float)
@click.option('--display_image_height', default=1000, help='Displayed image height in pixels. Image width will be calculated to keep the same ratio as the original capture source.', type=int)
@click.option('--df_export', is_flag=True, help='Export dataframe as a pickle file to the same directory as the video source.')
def main(src, model, diameter, detection_treshold, display_image_height, df_export):
    """
    Visualize the object detection model for barbell tracking on a video
    and create a dataframe containing the detected objects their raw
    and filtered positions and velocities at specific times in the video. 
    """
    for s in src:
        if not os.path.isfile(s):
            raise FileNotFoundError()

        # Initialize the tflite interpreter
        interpreter = tf.lite.Interpreter(model_path=model, num_threads=16)
        interpreter.allocate_tensors()

        data = track(s, interpreter, diameter,
                     detection_treshold, display_image_height)

        # FIXME: Replace idX with the id of the right object.
        if df_export:
            model_name = os.path.basename(model).split('.')[0]
            df_path = f'{s.split(".")[0]}_idX_{model_name}.pkl'
            df = pd.DataFrame.from_dict(data)
            df.to_pickle(df_path)


def track(src, interpreter, diameter, detection_treshold, display_image_height):
    """
    Runs the model inference visualization and returns captured data.
    """
    # Gather data about the video
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2h_ratio = float(frame_width)/float(frame_height)
    display_image_width = int(display_image_height*w2h_ratio)

    # Initialize Kalman Filters for each tracking_id (in the app, one KF will be enough)
    kf_args = {'dt': 1/fps, 'std_acc': 1, 'xm_std': 0.01, 'ym_std': 0.01}
    kfs = {}

    # Initialize the moving average filters for each tracking_id
    mas = {}

    # Initialize tracking variables
    frame_count = 0
    data = {'id': [], 'time': [], 'x_raw': [], 'y_raw': [],
            'x_filtered': [], 'y_filtered': [], 'dx': [], 'dy': []}

    # Store bar paths, key is object's tracking id, values are lists tuples [x, y]
    # representing the original image coordinates.
    raw_bar_paths = {}
    filtered_bar_paths = {}

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
                    mas[tracking_id] = MovingAverage(window_size=fps*10)

                norm_diameter = mas[tracking_id].process(
                    calc_normalized_diameter(result['bounding_box'])
                )
                print(norm_diameter)

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
            img_resized = cv2.resize(
                img, (display_image_width, display_image_height))
            cv2.imshow("Plate Tracking", img_resized)

            # Stream Control
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return data


if __name__ == "__main__":
    main()
