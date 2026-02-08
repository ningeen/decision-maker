import tempfile
import unittest
from pathlib import Path

import storage
from models import Project, Result


class TestStorage(unittest.TestCase):
    def test_save_and_load_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage.DATA_DIR = Path(temp_dir)

            project = Project(
                name="Test Project",
                criteria=["Cost"],
                pairwise=[[1.0]],
                weights=[1.0],
                options=["Option A"],
                scores=[[7.5]],
                results=[Result(option="Option A", score=0.75)],
            )

            path = storage.save_project(project)
            self.assertTrue(path.exists())

            loaded = storage.load_project("Test Project")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.name, "Test Project")
            self.assertEqual(loaded.criteria, ["Cost"])
            self.assertEqual(loaded.weights, [1.0])
            self.assertEqual(loaded.options, ["Option A"])
            self.assertEqual(loaded.scores, [[7.5]])
            self.assertEqual(loaded.results[0].option, "Option A")
            self.assertEqual(loaded.results[0].score, 0.75)
