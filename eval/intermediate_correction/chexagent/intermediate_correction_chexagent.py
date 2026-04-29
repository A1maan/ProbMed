import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXPERIMENT_DIR)
sys.path.append(EXPERIMENT_DIR)

from intermediate_layer_correction import main


if __name__ == "__main__":
    main(
        default_model="chexagent",
        default_results_file="../response_file/chexagent.json",
        default_test_file="../../../test/test.json",
        default_image_folder="../../../test",
    )
