import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(EXPERIMENT_DIR)
sys.path.append(EXPERIMENT_DIR)

try:
    import transformers.dynamic_module_utils as _dmu
    _orig_check = _dmu.check_imports
    def _patched_check(filename):
        try:
            return _orig_check(filename)
        except ImportError as e:
            if "tensorflow" in str(e):
                return _dmu.get_relative_imports(filename)
            raise
    _dmu.check_imports = _patched_check
except Exception:
    pass

from probe_direction_intervention.probe_direction_intervention import main

if __name__ == "__main__":
    main(
        default_model="medgemma",
        default_results_file="../response_file/medgemma.json",
        default_test_file="../../../test/test.json",
        default_image_folder="../../../test",
    )
