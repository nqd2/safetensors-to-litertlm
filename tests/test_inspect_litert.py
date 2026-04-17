import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from safetensors_to_litertlm.utils import model_inspector


class InspectLiteRTTests(unittest.TestCase):
    def test_resolve_report_path_keeps_relative_structure(self) -> None:
        report_path = model_inspector._resolve_report_path(Path("models/demo/model.tflite"))

        self.assertEqual(
            report_path,
            Path.cwd() / "inspections" / "models" / "demo" / "model-inspection.md",
        )

    def test_main_writes_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "models" / "demo" / "model.tflite"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"test-model")

            expected_report = root / "inspections" / "models" / "demo" / "model-inspection.md"
            stdout = io.StringIO()

            with patch.object(
                model_inspector,
                "inspect_litert_model",
                side_effect=lambda _: print("captured output"),
            ):
                with redirect_stdout(stdout):
                    with patch("pathlib.Path.cwd", return_value=root):
                        model_inspector.main([str(model_path)])

            self.assertIn("[*] Done!", stdout.getvalue())
            self.assertTrue(expected_report.exists())
            report = expected_report.read_text(encoding="utf-8")
            self.assertIn("# Inspection Report: `model`", report)
            self.assertIn(f"**Source:** `{model_path}`", report)
            self.assertIn("```text", report)
            self.assertIn("captured output", report)
            self.assertTrue(report.rstrip().endswith("```"))


if __name__ == "__main__":
    unittest.main()
