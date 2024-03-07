import cv2
import numpy as np
import tensorflow as tf

from math import sqrt

# Better performance if GPU's not used for tflite inference
tf.config.set_visible_devices([], 'GPU')


def preprocess_image(frame, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    original_img = tf.convert_to_tensor(frame, dtype=tf.dtypes.uint8)
    resized_img = tf.image.resize(original_img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_img


def calc_plate_width(bounding_box):
    """
    Calculate the width of a plate based on the
    detected bounding box in normalized image coordinates.
    """
    _, xmin, _, xmax = bounding_box

    return abs(xmax - xmin)


def calc_plate_height(bounding_box):
    """
    Calculate the height of a plate based on the
    detected bounding box in normalized image coordinates.
    """
    ymin, _, ymax, _ = bounding_box

    return abs(ymax - ymin)


def calc_bounding_box_center(bounding_box):
    """Calculates the center of a bounding box."""
    ymin, xmin, ymax, xmax = bounding_box
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    return center


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    boxes = np.squeeze(output['output_3'])

    results = []

    for i in range(count):
        if scores[i] >= threshold:
            results.append({
                'bounding_box': boxes[i],
                'score': scores[i]
            })

    return results


def run_odt(frame, interpreter, threshold=0.5):
    """Run object detection on the input image."""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, _ = preprocess_image(
        frame,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(
        interpreter, preprocessed_image, threshold=threshold)

    return results

def results_to_sorttracker_inputs(orig_results):
    """
    Transform the results into a format
    accepted by the SortTracker.
    """

    results = []

    # Tranform each result into a numpy array in form of [x1,y1,x2,y2,score,class]
    # where x1,y1 is the top left and x2,y2 is the bottom right.
    for res in orig_results:
        ymin, xmin, ymax, xmax = res['bounding_box']
        score = res['score']

        results.append(np.array([xmin, ymin, xmax, ymax, score, 0]))

    return np.empty((0,5)) if len(results) == 0 else np.array(results)
    

def draw_bounding_box(image, tracking_id, bounding_box, score, color):
    """Draw a bounding box based on the passed in results object."""
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = bounding_box
    xmin = int(xmin * image.shape[1])
    xmax = int(xmax * image.shape[1])
    ymin = int(ymin * image.shape[0])
    ymax = int(ymax * image.shape[0])

    # Draw the bounding box and label on the image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{:.0f}%, tracking_id: {}".format(
        score * 100, tracking_id)
    cv2.putText(image, label, (xmin, y),
                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


def draw_bar_path(image, bar_path, color):
    """Draw a bar path based on the passed in list of coordinates."""
    if len(bar_path) > 120:
        bar_path = bar_path[-120:]

    cv2.polylines(image, [bar_path], isClosed=False, color=color, thickness=2)
    cv2.circle(image, center=bar_path[-1],
               radius=10, color=color, thickness=-1)
