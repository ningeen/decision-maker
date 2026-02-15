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
                subcriteria=[["Upfront", "Ongoing"]],
                sub_pairwise=[[[1.0, 2.0], [0.5, 1.0]]],
                sub_weights=[[0.3, 0.7]],
                sub_consistency_ratio=[0.05],
                promethee_weights=[0.7],
                sub_promethee_weights=[[0.2, 0.8]],
                promethee_functions=["t3"],
                promethee_q=[0.0],
                promethee_p=[2.0],
                promethee_s=[0.0],
                promethee_directions=["min"],
                bwm_best_index=0,
                bwm_worst_index=0,
                bwm_best_to_others=[1.0],
                bwm_others_to_worst=[1.0],
                sub_bwm_best_index=[0],
                sub_bwm_worst_index=[1],
                sub_bwm_best_to_others=[[1.0, 3.0]],
                sub_bwm_others_to_worst=[[3.0, 1.0]],
                options=["Option A"],
                scores=[[7.5]],
                sub_scores=[[[7.0, 3.0]]],
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
            self.assertEqual(loaded.subcriteria, [["Upfront", "Ongoing"]])
            self.assertEqual(loaded.sub_pairwise, [[[1.0, 2.0], [0.5, 1.0]]])
            self.assertEqual(loaded.sub_weights, [[0.3, 0.7]])
            self.assertEqual(loaded.sub_consistency_ratio, [0.05])
            self.assertEqual(loaded.promethee_weights, [0.7])
            self.assertEqual(loaded.sub_promethee_weights, [[0.2, 0.8]])
            self.assertEqual(loaded.promethee_functions, ["t3"])
            self.assertEqual(loaded.promethee_q, [0.0])
            self.assertEqual(loaded.promethee_p, [2.0])
            self.assertEqual(loaded.promethee_s, [0.0])
            self.assertEqual(loaded.promethee_directions, ["min"])
            self.assertEqual(loaded.bwm_best_index, 0)
            self.assertEqual(loaded.bwm_worst_index, 0)
            self.assertEqual(loaded.bwm_best_to_others, [1.0])
            self.assertEqual(loaded.bwm_others_to_worst, [1.0])
            self.assertEqual(loaded.sub_bwm_best_index, [0])
            self.assertEqual(loaded.sub_bwm_worst_index, [1])
            self.assertEqual(loaded.sub_bwm_best_to_others, [[1.0, 3.0]])
            self.assertEqual(loaded.sub_bwm_others_to_worst, [[3.0, 1.0]])
            self.assertEqual(loaded.options, ["Option A"])
            self.assertEqual(loaded.scores, [[7.5]])
            self.assertEqual(loaded.sub_scores, [[[7.0, 3.0]]])
            self.assertEqual(loaded.results[0].option, "Option A")
            self.assertEqual(loaded.results[0].score, 0.75)
