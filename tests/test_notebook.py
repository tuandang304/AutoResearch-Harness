import ast
import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_embedded_sources_match_and_cells_parse(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/colab_gpu_executor.ipynb").read_text())
        embedded = set()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            if source.startswith("%%writefile"):
                directive, source = source.split("\n", 1)
                name = Path(directive.split()[1]).name
                self.assertEqual(
                    source.strip(),
                    (root / "ai_scientist/remote" / name).read_text().strip(),
                )
                embedded.add(name)
            if not source.startswith("!"):
                ast.parse(source)
        self.assertEqual(embedded, {"colab_server.py", "workspace.py"})


if __name__ == "__main__":
    unittest.main()
