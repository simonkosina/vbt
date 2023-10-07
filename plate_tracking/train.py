import os

from tflite_model_maker import object_detector
from tflite_model_maker import model_spec

# ARCHITECTURE = 'efficientdet_lite3' 
ARCHITECTURE = 'efficientdet_lite4' 
BATCH_SIZE = 8

if __name__ == "__main__":
    spec = model_spec.get(ARCHITECTURE)

    # TODO: Upload the dataset somewhere.
    train = object_detector.DataLoader.from_pascal_voc(
        images_dir="data/train",
        annotations_dir="data/train",
        label_map={1: "barbell"}
    )

    valid = object_detector.DataLoader.from_pascal_voc(
        images_dir="data/valid",
        annotations_dir="data/valid",
        label_map={1: "barbell"}
    )

    test = object_detector.DataLoader.from_pascal_voc(
        images_dir="data/test",
        annotations_dir="data/test",
        label_map={1: "barbell"}
    )

    model = object_detector.create(train, epochs=50, model_spec=spec, batch_size=BATCH_SIZE, validation_data=valid)

    print("Evaluating the original model...")
    print(model.evaluate(test, batch_size=BATCH_SIZE))

    print("Exporting the model...")
    model.export(export_dir='model', tflite_filename=f'{ARCHITECTURE}.tflite')

    print("Evaluating the exported model...")
    print(model.evaluate_tflite(f'model/{ARCHITECTURE}.tflite', test))
