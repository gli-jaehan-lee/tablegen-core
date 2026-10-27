"""Adversarial release checks, without numerical imports or network access."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('check_release', Path(__file__).resolve().parents[1] / 'tools/check_release.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReleaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', '.gitignore']:
            (self.root / name).write_text('Portable package\n')
        for name in ['tablegen', 'tablegen_addons', 'tablegen_extensions']:
            directory = self.root / 'src' / name
            directory.mkdir(parents=True)
            (directory / '__init__.py').write_text('')

    def issues(self):
        return checker.check_release(self.root, 'full')[0]

    def test_clean_tree(self):
        self.assertEqual(self.issues(), [])

    def test_missing_optional_namespace_is_caught(self):
        (self.root / 'src/tablegen_extensions/__init__.py').unlink()
        self.assertTrue(any('tablegen_extensions' in x for x in self.issues()))

    def test_private_path_and_secret_are_not_echoed(self):
        private = '/' + 'Users' + '/exampleperson/private'
        token = 'gh' + 'p_' + 'a' * 36
        (self.root / 'README.md').write_text(private + '\n' + token)
        result = self.issues()
        self.assertTrue(any('personal absolute path' in x for x in result))
        self.assertTrue(any('secret-like value' in x for x in result))
        self.assertNotIn(token, str(result))

    def test_symlink_and_env_are_caught(self):
        (self.root / 'linked').symlink_to(self.root / 'README.md')
        (self.root / '.env').write_text('SAMPLE=placeholder')
        self.assertTrue(any('Symlink' in x for x in self.issues()))
        self.assertTrue(any('.env' in x for x in self.issues()))

    def test_internal_docs_are_caught(self):
        (self.root / 'README.md').write_text('See ' + '_' + 'meta/' + 'PRIVATE.md')
        self.assertTrue(any('Internal documentation' in x for x in self.issues()))

    def test_core_roster_cannot_be_missing(self):
        self.assertIn('Core prior_index.json missing or malformed', checker.check_release(self.root, 'core')[0])


if __name__ == '__main__':
    unittest.main()
