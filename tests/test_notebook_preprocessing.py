import json
import unittest
from pathlib import Path


NOTEBOOKS = [
    Path("notebooks/sp500_daily.ipynb"),
    Path("notebooks/sp500_hourly.ipynb"),
]


def notebook_source(path):
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class NotebookPreprocessingTest(unittest.TestCase):
    def test_notebook_cell_schema_is_valid(self):
        for path in NOTEBOOKS:
            with self.subTest(path=str(path)):
                notebook = json.loads(path.read_text())

                for index, cell in enumerate(notebook["cells"]):
                    if cell.get("cell_type") == "markdown":
                        self.assertNotIn("outputs", cell, msg=f"markdown cell {index}")
                        self.assertNotIn("execution_count", cell, msg=f"markdown cell {index}")
                    elif cell.get("cell_type") == "code":
                        self.assertIn("outputs", cell, msg=f"code cell {index}")
                        self.assertIn("execution_count", cell, msg=f"code cell {index}")
                        self.assertEqual([], cell["outputs"], msg=f"code cell {index}")
                        self.assertIsNone(cell["execution_count"], msg=f"code cell {index}")

    def test_xlstm_notebooks_split_before_scaling(self):
        for path in NOTEBOOKS:
            with self.subTest(path=str(path)):
                source = notebook_source(path)

                self.assertIn("normalise_feature_target_split_data_xlstm", source)
                self.assertIn("create_feature_target_sequences", source)
                self.assertNotIn("normalise_data_xlstm", source)
                self.assertNotIn("close_scaled", source)
                self.assertNotIn("close_scaled_denoised", source)
                self.assertNotIn('STOCK, "Denoised", train_denoised', source)
                self.assertNotIn("show_results(final_results, 'Denoised')", source)
                self.assertIn("Causal Denoised", source)
                self.assertIn("repo_branch = 'fix/leakage-safe-denoising'", source)
                self.assertIn("git -C {repo_dir} pull --ff-only", source)
                self.assertIn("requirements_marker", source)
                self.assertIn("pd.read_csv(file_path", source)
                self.assertIn("str.slice(0, 19)", source)
                self.assertIn("if df.empty:", source)


if __name__ == "__main__":
    unittest.main()
