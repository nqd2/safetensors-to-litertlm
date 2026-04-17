import unittest

from safetensors_to_litertlm.converter import export


class ExportCliTests(unittest.TestCase):
    def test_extract_backend_default(self) -> None:
        backend, remaining = export._extract_backend(["--model-path", "m", "--output-dir", "o"])
        self.assertEqual(backend, "gemma4")
        self.assertEqual(remaining, ["--model-path", "m", "--output-dir", "o"])

    def test_extract_backend_explicit(self) -> None:
        backend, remaining = export._extract_backend(
            ["--backend", "gemma4", "--model-path", "m", "--output-dir", "o"]
        )
        self.assertEqual(backend, "gemma4")
        self.assertEqual(remaining, ["--model-path", "m", "--output-dir", "o"])


if __name__ == "__main__":
    unittest.main()
