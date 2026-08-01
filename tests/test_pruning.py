import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from virel_budget.backends.fastv_llava import FastVLlavaBackend
from virel_budget.pruning import select_tokens
from virel_budget.images import materialize_interventions
from virel_budget.schema import Sample


class PruningTest(unittest.TestCase):
    def test_full_budget_keeps_all_tokens(self):
        selection = select_tokens("random", "full", 16, sample_id="a", seed=1)
        self.assertEqual(selection.selected_indices, tuple(range(16)))

    def test_random_budget_is_deterministic(self):
        first = select_tokens("random", 4, 16, sample_id="a", seed=1)
        second = select_tokens("random", 4, 16, sample_id="a", seed=1)
        self.assertEqual(first.selected_indices, second.selected_indices)
        self.assertEqual(len(first.selected_indices), 4)

    def test_fastv_backend_configures_real_random_control(self):
        inner = SimpleNamespace(fast_v_selection_mode="attention", reset_fastv=lambda: None)
        config = SimpleNamespace()
        backend = object.__new__(FastVLlavaBackend)
        backend.runtime = SimpleNamespace(model=SimpleNamespace(model=inner, config=config))
        backend.dense_token_count = 576
        backend.fastv_agg_layer = 2
        backend._configure_fastv("random", 128, image_token_start=35, random_seed=991)
        self.assertTrue(config.use_fast_v)
        self.assertEqual(config.fast_v_attention_rank, 128)
        self.assertEqual(config.fast_v_selection_mode, "random")
        self.assertEqual(config.fast_v_random_seed, 991)

    def test_random_control_rejects_unpatched_fastv(self):
        backend = object.__new__(FastVLlavaBackend)
        backend.runtime = SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(), config=SimpleNamespace())
        )
        backend.dense_token_count = 576
        backend.fastv_agg_layer = 2
        with self.assertRaisesRegex(RuntimeError, "random_pruning.patch"):
            backend._configure_fastv("random", 64, image_token_start=35, random_seed=13)

    def test_center_budget_keeps_middle_cells(self):
        selection = select_tokens("center", 4, 16)
        self.assertEqual(len(selection.selected_indices), 4)
        self.assertEqual(selection.selected_indices, (5, 6, 9, 10))

    def test_gray_null_and_grayscale_are_distinct_interventions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (2, 1), color=(255, 0, 0)).save(source)
            sample = Sample(
                sample_id="sample",
                split="test",
                dataset="synthetic",
                image_path=source,
                question="What color?",
                answer="red",
            )
            variants = materialize_interventions(
                [sample],
                [
                    {"name": "gray_null", "type": "gray"},
                    {"name": "grayscale", "type": "grayscale"},
                ],
                root / "out",
                seed=13,
            )
            gray_null = Image.open(variants[("sample", "gray_null")].path).convert("RGB")
            grayscale = Image.open(variants[("sample", "grayscale")].path).convert("RGB")
            self.assertEqual(gray_null.getpixel((0, 0)), (128, 128, 128))
            self.assertNotEqual(grayscale.getpixel((0, 0)), (128, 128, 128))
            self.assertEqual(len(set(grayscale.getpixel((0, 0)))), 1)


if __name__ == "__main__":
    unittest.main()
