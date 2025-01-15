import unittest
import utils
from utils import get_dir_path
from ipydex import IPS


class TestCases1(unittest.TestCase):

    def test_001_read_conf(self):
        conf = utils.read_conf_from_dotenv()
        self.assertIsNotNone(conf.BASE_DIR)

    def test_002_get_dir_path(self):
        conf = utils.read_conf_from_dotenv()
        DATASET = "atsds_large"
        DATASET_SPLIT = "test"

        GROUND_TRUTH_DIR = get_dir_path(conf.BASE_DIR, f"{DATASET}_mask", DATASET_SPLIT)
        BACKGROUND_DIR = get_dir_path(conf.BASE_DIR, f"{DATASET}_background", DATASET_SPLIT)

        MODEL_TYPE = "convnext_tiny"
        XAI_TYPE = "gradcam"  # XAI method being applied
        # DATASET_DIR = get_dir_path(conf.BASE_DIR, DATASET, DATASET_SPLIT)
        # XAI_DIR = get_dir_path(conf.BASE_DIR, "auswertung_hpc", "auswertung", MODEL_TYPE, XAI_TYPE, DATASET_SPLIT)
        # ADV_FOLDER = get_dir_path(conf.BASE_DIR, "auswertung_hpc", "auswertung", MODEL_TYPE, XAI_TYPE, "adversarial")
