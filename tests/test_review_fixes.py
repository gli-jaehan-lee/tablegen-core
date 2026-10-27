import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from tablegen_core import GenerationConfig, generate


class ReviewFixTests(unittest.TestCase):
    def test_default_weight_alias_pool_order(self):
        expected=None
        for prior in ('default','default8'):
            for key in ('default','default8'):
                cfg,_=GenerationConfig(mode='dag',priors=[prior],prior_weights={key:2}).validated()
                if expected is None:expected=cfg.fct_types
                self.assertEqual(cfg.fct_types,expected)

    def test_concurrent_directory_creation_is_not_overwritten(self):
        result=generate(rows=8,features=2,priors=['lin'])
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'target';original=Path.mkdir
            def mkdir(path,*args,**kwargs):
                if path==target and not path.exists():original(path)
                return original(path,*args,**kwargs)
            with patch.object(Path,'mkdir',mkdir):
                with self.assertRaises(FileExistsError):result.write(target)
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()),[])
            self.assertFalse(list(Path(tmp).glob('.tablegen-core-*')))
