import cv2
import numpy as np

from RepCounter import RepCounter


def calculate_distance(landmark1, landmark2):
    x2 = (landmark1.x - landmark2.x)**2
    y2 = (landmark1.y - landmark2.y)**2
    z2 = (landmark1.z - landmark2.z)**2

    return (x2 + y2 + z2)**(1/2)


def height_plot(x_list, y_list, y_min, y_max, im_width, im_height=150, vpadding_px=10):
    y_size = im_height - vpadding_px
    im_graph = np.zeros((im_height, im_width, 3), dtype=np.uint8)

    if len(y_list) < 2:
        return im_graph

    xs = np.array(x_list[-im_width:], dtype=np.int32)
    ys = np.array(y_list[-im_width:], dtype=np.float32)

    # Adjust x coordinate when data starts to overflow
    if len(x_list) - im_width > 0:
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
    min_treshold = np.array([[0, min_scaled], [im_width, min_scaled]], dtype=np.int32)
    max_treshold = np.array([[0, max_scaled], [im_width, max_scaled]], dtype=np.int32)

    cv2.polylines(im_graph, [min_treshold], False, (0, 255, 0))  # Green
    cv2.polylines(im_graph, [max_treshold], False, (0, 0, 255))  # Red

    return im_graph
