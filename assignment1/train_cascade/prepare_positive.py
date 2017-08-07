import os
import uuid
import shutil
import sys
from PIL import Image
from train_cascade.config import *


def prepare_positive():
    print("[INFO] Started preparing positive samples...")

    # create output directory if not existed
    if not os.path.exists(OUTPUT_POSITIVE_DIR):
        os.makedirs(OUTPUT_POSITIVE_DIR)

    # empty output directory
    for file_object in os.listdir(OUTPUT_POSITIVE_DIR):
        file_object_path = os.path.join(OUTPUT_POSITIVE_DIR, file_object)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)

    for filename in os.listdir(INPUT_POSITIVE_DIR):
        input_path = os.path.join(INPUT_POSITIVE_DIR, filename)

        if os.path.isdir(input_path):
            continue

        new_filename = filename
        if " " in filename:
            _, file_extension = os.path.splitext(filename)
            new_filename = str(uuid.uuid4()).replace("-", "") + file_extension

        output_path = os.path.join(OUTPUT_POSITIVE_DIR, new_filename)
        shutil.copy2(input_path, output_path)

    # Create the positive.txt input file
    f = open(POSITIVE_INFO_FILE, 'w')
    count = 0
    for filename in os.listdir(OUTPUT_POSITIVE_DIR):
        if os.path.isdir(OUTPUT_POSITIVE_DIR + filename):
            continue

        if filename.endswith(".txt"):
            continue

        if filename.endswith(".db"):
            continue

        try:
            img = Image.open(os.path.join(OUTPUT_POSITIVE_DIR, filename))

            # get the image's width and height in pixels
            width, height = img.size
            f.write(filename + " 1 0 0 " + str(width) +
                    " " + str(height) + '\n')
            count += 1

        except IOError:
            print("Exception reading image file: {}".format(filename))

    f.close()

    with open(NUM_POS_FILE, "w") as f:
        f.write(str(count))

    print("[INFO] Finished.")


if __name__ == "__main__":
    prepare_positive()
