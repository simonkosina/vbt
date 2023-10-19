import cv2
import numpy as np
import tensorflow as tf

from math import sqrt

# Better if GPU's not used for tflite inference
tf.config.set_visible_devices([], 'GPU')

# Define a list of colors for visualization
COLORS = [[255, 0, 255]]


def preprocess_image(frame, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    original_img = tf.convert_to_tensor(frame, dtype=tf.dtypes.uint8)
    resized_img = tf.image.resize(original_img, input_size)
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


def run_odt(frame, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, _ = preprocess_image(
        frame,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results, classes = detect_objects(
        interpreter, preprocessed_image, threshold=threshold)

    return results


def draw_results(image, results, bar_paths):
    # Plot the detection results on the input image
    original_image_np = np.array(image, dtype=np.uint8)

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
