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
                promethee_weights=[0.7],
                promethee_functions=["t3"],
                promethee_q=[0.0],
                promethee_p=[2.0],
                promethee_s=[0.0],
                promethee_directions=["min"],
                bwm_best_index=0,
                bwm_worst_index=0,
                bwm_best_to_others=[1.0],
                bwm_others_to_worst=[1.0],
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
            self.assertEqual(loaded.promethee_weights, [0.7])
            self.assertEqual(loaded.promethee_functions, ["t3"])
            self.assertEqual(loaded.promethee_q, [0.0])
            self.assertEqual(loaded.promethee_p, [2.0])
            self.assertEqual(loaded.promethee_s, [0.0])
            self.assertEqual(loaded.promethee_directions, ["min"])
            self.assertEqual(loaded.bwm_best_index, 0)
            self.assertEqual(loaded.bwm_worst_index, 0)
            self.assertEqual(loaded.bwm_best_to_others, [1.0])
            self.assertEqual(loaded.bwm_others_to_worst, [1.0])
            self.assertEqual(loaded.options, ["Option A"])
            self.assertEqual(loaded.scores, [[7.5]])
            self.assertEqual(loaded.results[0].option, "Option A")
            self.assertEqual(loaded.results[0].score, 0.75)
