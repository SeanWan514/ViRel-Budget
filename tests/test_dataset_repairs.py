import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from virel_budget.datasets.jsonl import load_jsonl_samples, write_jsonl
from virel_budget.images import materialize_interventions


class DatasetRepairTest(unittest.TestCase):
    def test_mmstar_letter_options_are_repaired_to_full_text(self):
        root = Path(tempfile.mkdtemp(prefix="virel-mmstar-"))
        try:
            img = root / "image.png"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img)
            row = {
                "sample_id": "mmstar_test",
                "split": "test",
                "dataset": "mmstar",
                "image_path": "image.png",
                "question": "What is shown?\nOptions: A: A red square, B: A blue circle, C: A green bar, D: A white dot",
                "answer": "A",
                "options": ["A", "B", "C", "D"],
                "metadata": {},
            }
            write_jsonl(root / "samples.jsonl", [row])
            sample = load_jsonl_samples(root / "samples.jsonl", "mmstar")[0]
            self.assertEqual(sample.answer, "A red square")
            self.assertEqual(sample.options[0], "A red square")
            self.assertEqual(sample.metadata["answer_letter"], "A")
            self.assertEqual(sample.metadata["option_map"]["B"], "A blue circle")
        finally:
            shutil.rmtree(root)

    def test_mmstar_full_text_options_keep_option_map(self):
        root = Path(tempfile.mkdtemp(prefix="virel-mmstar-map-"))
        try:
            img = root / "image.png"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img)
            row = {
                "sample_id": "mmstar_text_options",
                "split": "test",
                "dataset": "mmstar",
                "image_path": "image.png",
                "question": "What is shown?\nOptions: A: A red square, B: A blue circle, C: A green bar, D: A white dot",
                "answer": "A red square",
                "options": ["A red square", "A blue circle", "A green bar", "A white dot"],
                "metadata": {"source_answer": "A"},
            }
            write_jsonl(root / "samples.jsonl", [row])
            sample = load_jsonl_samples(root / "samples.jsonl", "mmstar")[0]
            self.assertEqual(sample.answer, "A red square")
            self.assertEqual(sample.options[1], "A blue circle")
            self.assertEqual(sample.metadata["option_map"]["B"], "A blue circle")
        finally:
            shutil.rmtree(root)

    def test_mmstar_choices_format_keeps_option_map(self):
        root = Path(tempfile.mkdtemp(prefix="virel-mmstar-choices-"))
        try:
            img = root / "image.png"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img)
            row = {
                "sample_id": "mmstar_choices",
                "split": "test",
                "dataset": "mmstar",
                "image_path": "image.png",
                "question": "Question: What value?\nChoices:\n(A) 4\n(B) 5\n(C) 6\n(D) 7",
                "answer": "6",
                "options": ["4", "5", "6", "7"],
                "metadata": {"source_answer": "C"},
            }
            write_jsonl(root / "samples.jsonl", [row])
            sample = load_jsonl_samples(root / "samples.jsonl", "mmstar")[0]
            self.assertEqual(sample.answer, "6")
            self.assertEqual(sample.metadata["option_map"]["C"], "6")
        finally:
            shutil.rmtree(root)

    def test_counterfactual_intervention_is_skipped_when_unavailable(self):
        root = Path(tempfile.mkdtemp(prefix="virel-cf-"))
        try:
            img = root / "image.png"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img)
            row = {
                "sample_id": "plain",
                "split": "test",
                "dataset": "pope_random",
                "image_path": "image.png",
                "question": "Is there a square?",
                "answer": "yes",
                "options": ["yes", "no"],
            }
            write_jsonl(root / "samples.jsonl", [row])
            sample = load_jsonl_samples(root / "samples.jsonl", "pope_random")[0]
            interventions = materialize_interventions(
                [sample],
                [{"name": "counterfactual", "type": "counterfactual"}],
                root / "interventions",
                seed=13,
            )
            self.assertNotIn(("plain", "counterfactual"), interventions)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
