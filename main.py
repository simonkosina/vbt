import click
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import os

from sort.tracker import SortTracker
from KalmanFilter import KalmanFilter
from odt import run_odt, detect_objects, draw_bar_path, draw_bounding_box, calc_bounding_box_center, calc_plate_height, calc_plate_width, results_to_sorttracker_inputs
from tflite_runtime.interpreter import Interpreter


tf.config.set_visible_devices([], 'GPU')

MAX_AGE = 30
COLORS = [(115, 3, 252), (255, 255, 255)]

@click.command()
@click.argument('src', type=str, nargs=-1)
@click.option('--model', default='models/efficientdet_lite0_whole.tflite', help='Path to a TF Lite model used for object detection.', type=str)
@click.option('--detection_treshold', default=0.5, help='Object detection threshold.', type=float)
@click.option('--display_image_height', default=720, help='Displayed image height in pixels. Image width will be calculated to keep the same ratio as the original capture source.', type=int)
@click.option('--df_export', is_flag=True, help='Export dataframe as a pickle file to the same directory as the video source.')
@click.option('--df_dir', default=None, help='Directory for exporting the dataframes.')
@click.option('--video_export', is_flag=True, help='Export a video showing the tracked objects and bar path.')
@click.option('--video_dir', default=None, help='Directory for exporting the video with tracked objects and bar path.')
@click.option('--threads', default=4, help='Number of threads to use for detection model inference.')
def main(src, model, detection_treshold, display_image_height, df_export, df_dir, video_export, video_dir, threads):
    """
    Visualize the object detection model for barbell tracking on a video
    and create a dataframe containing the detected objects their raw
    and filtered positions and velocities at specific times in the video. 
    """
    if df_dir is not None:
        os.makedirs(df_dir, exist_ok=True)

    if video_dir is not None:
        os.makedirs(df_dir, exist_ok=True)

    for s in src:
        if not os.path.isfile(s):
            raise FileNotFoundError()

        # Initialize the tflite interpreter
        interpreter = Interpreter(model_path=model, num_threads=threads)
        interpreter.allocate_tensors()

        video_filename = f'{os.path.basename(s).split(".")[0]}.mp4'

        if video_dir is None:
            video_path = video_filename
        else:
            video_path = os.path.join(video_dir, video_filename)

        data = track(s, interpreter, detection_treshold, display_image_height, video_path if video_export else None)

        if df_export:
            df = pd.DataFrame.from_dict(data)
            df = df.sort_values(by=['id', 'time'])
            df2 = df.copy()

            # Calculate the Euclidean distances for each row
            df2['distance'] = np.where(df2['id'] == df2['id'].shift(), ((df2['x'] - df2['x'].shift())**2 + (df2['y'] - df2['y'].shift())**2)**0.5, np.nan)

            # Calculate the cumulative distance for each 'id'
            df2['cumulative_distance'] = df2.groupby('id')['distance'].cumsum()
            max_distance_id = df2.loc[df2['cumulative_distance'].idxmax(), 'id']

            model_name = os.path.basename(model).split('.')[0]
            df_filename = f'{os.path.basename(s).split(".")[0]}_id{max_distance_id}_{model_name}.pkl'

            if df_dir is None:
                df_path = df_filename
            else:
                df_path = os.path.join(df_dir, df_filename)

            if df_export:
                df.to_pickle(df_path)


def track(src, interpreter, detection_treshold, display_image_height, video_path):
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

    # Initialize tracking variables
    frame_count = 0
    data = {'id': [], 'time': [], 'x': [], 'y': [], 'dx': [], 'dy': [],
            'norm_plate_height': [], 'norm_plate_width': []}

    # Store bar paths, key is object's tracking id, values are lists tuples [x, y]
    # representing the original image coordinates.
    bar_paths = {}
        
    # Initialize video writer
    if video_path is not None:
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    tracker = SortTracker(max_age=MAX_AGE)

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break

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

        tracker_out = tracker.update(results_to_sorttracker_inputs(results), [])

        for res in tracker_out:
            xmin, ymin, xmax, ymax, tracking_id, _, score = res
            bounding_box = [ymin, xmin, ymax, xmax]
            tracking_id = int(tracking_id)

            for trk in tracker.trackers:
                if trk.id == tracking_id - 1:
                    kf = trk.kf
                    break

            dx, dy = kf.x.flatten()[4:6]

            draw_bounding_box(
                image=img,
                tracking_id=tracking_id,
                bounding_box=bounding_box,
                score=score,
                color=COLORS[1]
            )

            # Center in normalized image coordinates
            x_center, y_center = calc_bounding_box_center(bounding_box)

            # Bounding box center in image coordinates
            center_im = np.array(
                [x_center*img.shape[1], y_center*img.shape[0]], dtype=np.int32)

            # Store and draw the bar paths
            if tracking_id in bar_paths:
                bar_paths[tracking_id] = np.concatenate(
                    (bar_paths[tracking_id], [center_im]), dtype=np.int32)
            else:
                bar_paths[tracking_id] = np.array(
                    [center_im], np.int32)

            draw_bar_path(img, bar_paths[tracking_id], color=COLORS[1])

            # Append data to the dataframe
            data['id'].append(tracking_id)
            data['time'].append(time)
            data['x'].append(x_center)
            data['y'].append(y_center)
            data['dx'].append(dx)
            data['dy'].append(dy)
            data['norm_plate_height'].append(calc_plate_height(bounding_box))
            data['norm_plate_width'].append(calc_plate_width(bounding_box))

        # Show results
        img_resized = cv2.resize(
            img, (display_image_width, display_image_height))
        cv2.imshow(os.path.basename(src), img_resized)

        if video_path is not None:
            video_writer.write(img)

        # Stream Control
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if video_path is not None:
        video_writer.release()

    if hasattr(detect_objects, 'last_tracking_id'):
        delattr(detect_objects, 'last_tracking_id')
    if hasattr(detect_objects, 'prev_results'):
        delattr(detect_objects, 'prev_results')

    return data


if __name__ == "__main__":
    main()
