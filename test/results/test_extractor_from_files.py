import os
import numpy as np
from unittest import TestCase
from unittest.mock import patch
from marloes.results.extractor import Extractor


class TestExtractorFromFiles(TestCase):
    def setUp(self):
        """
        Set up the Extractor and mock data for testing from_files functionality.
        """
        self.extractor = Extractor(chunk_size=1000, from_model=False)

        # Mock data structure in results directory
        self.mock_uid = 42
        self.mock_results = {
            "test_results/solar": f"test_results/solar/solar_{self.mock_uid}.npy",
            "test_results/battery": f"test_results/battery/battery_{self.mock_uid}.npy",
            "test_results/grid": f"test_results/grid/grid_{self.mock_uid}.npy",
        }

        # Create mock files and directories
        for folder, file_path in self.mock_results.items():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, np.random.random(10))  # Save random data for testing

        # Also create a UID file (uid.txt) in the results directory
        with open("test_results/uid.txt", "w") as f:
            f.write(str(self.mock_uid + 1))

    def tearDown(self):
        """
        Clean up mock files and directories.
        """
        for folder in self.mock_results.keys():
            folder_path = os.path.dirname(folder)
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    os.rmdir(root)

    def test_from_files_valid_uid(self):
        """
        Test that Extractor.from_files loads data correctly when a valid UID is provided.
        """
        self.extractor.from_files(uid=self.mock_uid, dir="test_results")

        for folder, file_path in self.mock_results.items():
            attribute_name = os.path.basename(folder)
            expected_data = np.load(file_path)
            actual_data = getattr(self.extractor, attribute_name, None)

            self.assertIsNotNone(
                actual_data, f"Attribute '{attribute_name}' not found."
            )
            np.testing.assert_array_almost_equal(
                actual_data,
                expected_data,
                err_msg=f"Data for '{attribute_name}' does not match expected values.",
            )

    def test_from_files_latest_uid(self):
        """
        Test that Extractor.from_files uses the latest UID when none is provided.
        """
        with patch("numpy.load") as mock_load:
            self.extractor.from_files(uid=None, dir="test_results")

            # Check that the latest UID was used
            mock_load.assert_any_call(f"test_results/solar/solar_{self.mock_uid}.npy")

    def test_from_files_missing_file(self):
        """
        Test that Extractor.from_files skips folders without the matching UID file.
        """
        # Remove one of the files
        os.remove(self.mock_results["test_results/solar"])

        self.extractor.from_files(uid=self.mock_uid, dir="test_results")

        # Check that the solar attribute was not set
        self.assertFalse(
            hasattr(self.extractor, "solar"),
            "Extractor should skip folders with missing UID files.",
        )
