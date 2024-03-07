"""
Simple script to partition files exported from LabelStudio
to train. test and valid directories.
"""

import os
import glob
import numpy as np

ANNOTATIONS_DIR = "data/tmp/project3/Annotations"
IMAGES_DIR = "data/tmp/project3/images"
DEST_DIR = "data"

TRAIN_PERCENTAGE = 0.85
TEST_PERCENTAGE = 0.05
VALID_PERCENTAGE = 0.1

def copy_files(filenames, partition_dir):
    for filename in filenames:
        xml_src = os.path.join(ANNOTATIONS_DIR, filename) + ".xml"
        xml_dst = os.path.join(DEST_DIR, partition_dir, filename) + ".xml"

        jpg_src = os.path.join(IMAGES_DIR, filename) + ".jpg" 
        jpg_dst = os.path.join(DEST_DIR, partition_dir, filename) + ".jpg" 

        os.system(f"cp {xml_src} {xml_dst}")
        os.system(f"cp {jpg_src} {jpg_dst}")


if __name__ == "__main__":
    files = glob.glob(pathname=f"{ANNOTATIONS_DIR}/*")
    
    # Extract only filenames without the '.xml' extension
    files = list(map(lambda x: os.path.basename(x)[:-4], files))
    np.random.shuffle(files)

    num_train = round(len(files)*TRAIN_PERCENTAGE)
    num_test = round(len(files)*TEST_PERCENTAGE)
    num_valid = round(len(files)*VALID_PERCENTAGE)

    train = files[0:num_train]
    test = files[num_train:num_train+num_test]
    valid = files[-num_valid:]
    
    copy_files(train, "train")
    copy_files(test, "test")
    copy_files(valid, "valid")
