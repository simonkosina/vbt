# TODO: Cite https://www.tensorflow.org/lite/models/modify/model_maker/object_detection#run_object_detection_and_show_the_detection_results
# TODO: Can't let the plates be detected in different order. Let the user pick the right one, if multiple plates detected and based on distance, decide which is which.

import time
import cv2
import numpy as np
import tensorflow as tf

from math import sqrt

tf.config.set_visible_devices([], 'GPU')

# MODEL_PATH = "plate_tracking/models/efficientdet_lite0_whole.tflite"
# MODEL_PATH = "plate_tracking/models/efficientdet_lite1_whole.tflite"
MODEL_PATH = "plate_tracking/models/efficientdet_lite2_whole.tflite"
CAPTURE_SOURCE = "plate_tracking/samples/cut/003_squat_7_reps.mp4"
# CAPTURE_SOURCE = "plate_tracking/samples/cut/027_smith_bp_7_reps.mp4"
# CAPTURE_SOURCE = "plate_tracking/samples/cut/016_squat_8_reps.mp4"
IM_HEIGHT_PX = 1000
DETECTION_TRESHOLD = 0.5

# Define a list of colors for visualization
COLORS = [[255, 0, 255]]

bar_paths = {}


def preprocess_image(frame, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    original_img = tf.convert_to_tensor(frame, dtype=tf.dtypes.uint8)
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_img


def calc_bounding_box_center(box):
    """Calculates the center of a bounding box."""
    ymin, xmin, ymax, xmax = box
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    return center


def check_box_overlap_1d(a, b):
    """Check if there is overlap between two 1d bounding boxes."""
    a_xmin, a_xmax = a
    b_xmin, b_xmax = b

    return a_xmax >= b_xmin and a_xmin <= b_xmax


def check_box_overlap_2d(a, b):
    """Check if there is an overlap between two 2d bounding boxes."""
    a_ymin, a_xmin, a_ymax, a_xmax = a
    b_ymin, b_xmin, b_ymax, b_xmax = b

    overlap = check_box_overlap_1d((a_xmin, a_xmax), (b_xmin, b_xmax))
    overlap = overlap and check_box_overlap_1d(
        (a_ymin, a_ymax), (b_ymin, b_ymax))

    return overlap


def find_closest_object(box, prev_results):
    """
    Based on provided bounding boxes returns the id
    of the closest object in previous results and
    boolean value saying if the new and the previous
    bounding box overlaps.
    """
    best_distance = None
    best_id = None
    best_box = None
    center = calc_bounding_box_center(box)

    for tracking_id, result in prev_results.items():
        prev_center = calc_bounding_box_center(result['bounding_box'])
        distance = sqrt((prev_center[0] - center[0])
                        ** 2 + (prev_center[1] - center[1])**2)
        if best_distance is None or distance < best_distance:
            best_box = result['bounding_box']
            best_distance = distance
            best_id = tracking_id

    return best_id, check_box_overlap_2d(best_box, box)


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    if not hasattr(detect_objects, 'last_tracking_id'):
        detect_objects.last_tracking_id = 0

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = {}

    for i in range(count):
        if scores[i] >= threshold:
            if hasattr(detect_objects, "prev_results"):
                tracking_id, overlap = find_closest_object(
                    boxes[i], detect_objects.prev_results)
                if not overlap:
                    tracking_id = detect_objects.last_tracking_id
                    detect_objects.last_tracking_id += 1
            else:
                tracking_id = detect_objects.last_tracking_id
                detect_objects.last_tracking_id += 1

            results[tracking_id] = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }

    # Remember results to identify objects in next detection
    if hasattr(detect_objects, "prev_results"):
        detect_objects.prev_results.update(results)
    else:
        detect_objects.prev_results = results

    return results, classes


def run_odt_and_draw_results(frame, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        frame,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results, classes = detect_objects(
        interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for tracking_id, obj in results.items():
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Draw the bar path
        center = [round((xmin + xmax) / 2), round((ymin + ymax) / 2)]
        if tracking_id in bar_paths:
            bar_paths[tracking_id] = np.concatenate(
                (bar_paths[tracking_id], [center]), dtype=np.int32)
        else:
            bar_paths[tracking_id] = np.array([center], np.int32)
        cv2.polylines(original_image_np, [
                      bar_paths[tracking_id]], isClosed=False, color=color, thickness=2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{:.0f}%, tracking_id: {}".format(
            obj['score'] * 100, tracking_id)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)

    count = 0

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=16)
    interpreter.allocate_tensors()

    while (cap.isOpened()):
        ret, frame = cap.read()
        count = (count + 1) % 5

        if ret and not count:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = True

            img = run_odt_and_draw_results(
                img,
                interpreter,
                threshold=DETECTION_TRESHOLD
            )

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_resized = cv2.resize(img, (IM_WIDTH_PX, IM_HEIGHT_PX))

            cv2.imshow("Plate Tracking", img_resized)

            # Stream Control
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
