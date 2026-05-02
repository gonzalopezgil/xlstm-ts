import csv
import json
import unittest
from pathlib import Path


NOTEBOOKS = {
    Path("notebooks/sp500_daily.ipynb"): Path("data/results/xlstm_daily_leakage_safe_results.csv"),
    Path("notebooks/sp500_hourly.ipynb"): Path("data/results/xlstm_hourly_leakage_safe_results.csv"),
}

RENDERED_RESULTS_TAG = "xlstm_ts_rendered_results"

DISPLAY_LABELS = {
    "current_raw_xlstm_ts": "Current raw xLSTM-TS",
    "current_causal_denoised_feature_xlstm_ts": "Current causal-denoised-feature xLSTM-TS",
    "old_cached_raw_xlstm_ts": "Old cached raw xLSTM-TS",
    "old_cached_offline_denoised_xlstm_ts": "Old cached offline-denoised xLSTM-TS",
}


def notebook_source(path):
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def notebook_text(path):
    notebook = json.loads(path.read_text())
    return json.dumps(notebook, sort_keys=True)


def read_result_rows(path):
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def rendered_result_cells(notebook):
    return [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get(RENDERED_RESULTS_TAG)
    ]


def pct(value):
    return f"{float(value):.2f}%"


def metric(value):
    return f"{float(value):.2f}"


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
                        if cell.get("metadata", {}).get(RENDERED_RESULTS_TAG):
                            self.assertIsNotNone(cell["execution_count"], msg=f"code cell {index}")
                            self.assertGreater(len(cell["outputs"]), 0, msg=f"code cell {index}")
                        else:
                            self.assertEqual([], cell["outputs"], msg=f"code cell {index}")
                            self.assertIsNone(cell["execution_count"], msg=f"code cell {index}")

    def test_rendered_result_tables_match_committed_csvs(self):
        for notebook_path, result_path in NOTEBOOKS.items():
            with self.subTest(path=str(notebook_path)):
                notebook = json.loads(notebook_path.read_text())
                rendered_cells = rendered_result_cells(notebook)
                self.assertEqual(1, len(rendered_cells))

                rendered_cell = rendered_cells[0]
                rendered_text = notebook_text(notebook_path)
                outputs = rendered_cell["outputs"]
                self.assertTrue(
                    any("text/html" in output.get("data", {}) for output in outputs),
                    msg="rendered result cell must include an HTML table output",
                )

                rows = read_result_rows(result_path)
                self.assertEqual(4, len(rows))
                for row in rows:
                    label = DISPLAY_LABELS[row["pipeline"]]
                    self.assertIn(label, rendered_text)
                    self.assertIn(pct(row["test_accuracy"]), rendered_text)
                    self.assertIn(pct(row["f1_score"]), rendered_text)
                    self.assertIn(metric(row["mae"]), rendered_text)
                    self.assertIn(metric(row["rmse"]), rendered_text)

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
