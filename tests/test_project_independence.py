import unittest
from pathlib import Path

from utils.project_checks import find_symlinks, scan_for_forbidden_runtime_links


class ProjectIndependenceTest(unittest.TestCase):
    def test_no_forbidden_runtime_links_or_symlinks(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(scan_for_forbidden_runtime_links(root), [])
        self.assertEqual(find_symlinks(root), [])


if __name__ == "__main__":
    unittest.main()
