import os
import sys
from train_cascade.config import *
from datetime import datetime
import time
import shutil


def create_vecfile(width=WIDTH, height=HEIGHT, num=MAX_NUM_POS):
    print("[INFO] Started collapsing the samples into a vector file...")
    exec_str = "{} -vec {} -w {} -h {} -info {} -num {}".format(
        SAMPLE_CREATOR, VEC_FILE, width, height, POSITIVE_INFO_FILE, num)
    # print(exec_str)
    os.system(exec_str)
    print("[INFO] Finished.")


def train_cascade(output_dir=OUTPUT_DIR, 
                  width=WIDTH, 
                  height=HEIGHT, 
                  num_pos=NUM_POS, 
                  num_neg=NUM_NEG, 
                  max_far=MAX_FALSE_ALARM_RATE, 
                  num_stages=NUM_STAGES,
                  clear_previous_output=False):
    if clear_previous_output:
        if os.path.isdir(output_dir):
            print("[INFO] Cleared previous output.")
            shutil.rmtree(output_dir)
            
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print("[INFO] Started training cascade...")
    exec_str = "{} -data {} -vec {} -bg {} -w {} -h {} -numPos {} -numNeg {} \
                -maxFalseAlarmRate {} -featureType {} -numStages {} -minHitRate {} \
                -bt {} {}".format(
        CASCADE_TRAINER, output_dir, VEC_FILE, NEGATIVE_INFO_FILE, width,
        height, num_pos, num_neg, max_far, FEATURE_TYPE, num_stages, MIN_HIT_RATE, BOOST_TYPE,
        EXTRA_PARAMS)
    # print(exec_str)
    os.system(exec_str)
    print("[INFO] Finished.")


if __name__ == "__main__":
    train_cascade()