import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXPERIMENT_DIR)
sys.path.append(EXPERIMENT_DIR)

from train_vector_correction import main

if __name__ == "__main__":
    main(
        default_model="llavamed",
        default_results_file="../response_file/llavamed.json",
        default_test_file="../../../test/test.json",
        default_image_folder="../../../test",
        default_load_8bit=False,  # 8-bit blocks backprop — load in fp16
    )
