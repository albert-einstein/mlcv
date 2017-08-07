import os
import shutil
import sys
from train_cascade.config import *


def prepare_negative():
    print("[INFO] Started preparing negative samples...")

    # create output directory if not existed
    if not os.path.exists(OUTPUT_NEGATIVE_DIR):
        os.makedirs(OUTPUT_NEGATIVE_DIR)

    # empty output directory
    for file_object in os.listdir(OUTPUT_NEGATIVE_DIR):
        file_object_path = os.path.join(OUTPUT_NEGATIVE_DIR, file_object)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)

    count = 0
    with open(NEGATIVE_INFO_FILE, "w") as f:
        # write a list of all the negative files
        for filename in os.listdir(INPUT_NEGATIVE_DIR):
            input_path = os.path.join(INPUT_NEGATIVE_DIR, filename)

            if os.path.isdir(input_path):
                continue
                
            if filename.endswith(".txt"):
                continue

            if filename.endswith(".db"):
                continue

            output_path = os.path.join(OUTPUT_NEGATIVE_DIR, filename)
            shutil.copy2(input_path, output_path)
            f.write(os.path.abspath(output_path) + "\n")
            count += 1

    with open(NUM_NEG_FILE, "w") as f:
        f.write(str(count))

    print("[INFO] Finished.")

if __name__ == "__main__":
    prepare_negative()
