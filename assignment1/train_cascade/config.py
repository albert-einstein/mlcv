import os


OUTPUT_DIR = "data/new_cascade/"

###################### change parameters ########################
WIDTH = 20
HEIGHT = 20
MAX_NUM_POS = 100

NUM_POS = 90
NUM_NEG = 100
NUM_STAGES = 7

MAX_FALSE_ALARM_RATE = 0.5
MIN_HIT_RATE = 0.995

FEATURE_TYPE = "LBP"
BOOST_TYPE = "DAB"

EXTRA_PARAMS = "-precalcValBufSize 1000 -precalcIdxBufSize 1000"

#################################################################

OPENCV_DIR = ""

BASE_DIR = "data/"
BATCH_FILE = os.path.join(BASE_DIR, "batch.txt")
SAMPLE_CREATOR = os.path.join(OPENCV_DIR, "opencv_createsamples")
CASCADE_TRAINER = os.path.join(OPENCV_DIR, "opencv_traincascade")

INPUT_NEGATIVE_DIR = os.path.join(BASE_DIR, "input_negative")
INPUT_POSITIVE_DIR = os.path.join(BASE_DIR, "input_positive")

OUTPUT_NEGATIVE_DIR = os.path.join(BASE_DIR, "output_negative")
OUTPUT_POSITIVE_DIR = os.path.join(BASE_DIR, "output_positive")

NEGATIVE_INFO_FILE = os.path.join(OUTPUT_NEGATIVE_DIR, "negative.txt")
NUM_NEG_FILE = os.path.join(OUTPUT_NEGATIVE_DIR, "numNeg.txt")
POSITIVE_INFO_FILE = os.path.join(OUTPUT_POSITIVE_DIR, "positive.txt")
NUM_POS_FILE = os.path.join(OUTPUT_POSITIVE_DIR, "numPos.txt")
VEC_FILE = os.path.join(OUTPUT_POSITIVE_DIR, "vecfile.vec")
