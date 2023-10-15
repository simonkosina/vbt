import cv2
import numpy as np
import tensorflow as tf

TFLITE_MODEL_PATH = "plate_tracking/models/efficientdet_lite2_whole.tflite"
CAPTURE_SOURCE = "plate_tracking/samples/squat_8reps_cut.mp4"
IM_HEIGHT_PX = 1200

if __name__ == "__main__":
    interpreter = tf.lite.Interpreter(TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(f"Input Shape: {input_shape}")

    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RATIO = float(frame_width)/float(frame_height)
    IM_WIDTH_PX = int(IM_HEIGHT_PX*RATIO)

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = True

            input_data = tf.convert_to_tensor(img, dtype=tf.dtypes.uint8)
            input_data = tf.image.resize(
                input_data,
                (448,448),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            input_data = tf.einput_datapand_dims(input_data, 0)

            interpreter.set_tensor(input_details[0]['indeinput_data'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_resized = cv2.resize(img, (IM_WIDTH_PX, IM_HEIGHT_PX))

            cv2.imshow("Plate Tracking", img_resized)

            # Stream Control
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            break
