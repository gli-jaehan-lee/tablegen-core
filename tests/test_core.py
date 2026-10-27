import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from tablegen_core import GenerationConfig, TableResult, generate, list_priors
from tablegen_core.registry import resolve, DEFAULT8

# Independent literal expectation, not derived from the registry under test.
EXPECTED = set('disc em gp lin mlp prod quad tree piecewise_route_engine nori_anchor_rho_copy gev_tail_warp ewens_sampling_formula markov_state_error_emission informative_censoring_hazard eos_excess_mixing_enthalpy modulated_satellite_structure_factor paws_timestamp_drop niche_model_food_web island_biogeography_equilibrium bourgouin_precipitation_type temperature_dependent_instar_transition tetragonal_interplanar_spacing bch_error_locator moe_linear_experts'.split())


class CoreTests(unittest.TestCase):
    def test_exact24_and_closed_imports(self):
        entries = list_priors()
        self.assertEqual(len(entries), 24)
        self.assertEqual({e['name'] for e in entries}, EXPECTED)
        self.assertEqual(len({e['id'] for e in entries}), 24)
        self.assertTrue(all(e['selectable'] for e in entries))
        for entry in entries:
            self.assertEqual(resolve(entry['id']), entry['name'])
        self.assertFalse(any(name == 'tablegen' or name.startswith('tablegen.') for name in sys.modules))
        imported = {name.rsplit('.', 1)[-1] for name in sys.modules if name.startswith('tablegen_core.priors.nodes.')}
        self.assertEqual(imported, EXPECTED - set('disc em gp lin mlp prod quad tree'.split()))

    def test_invalid_prior(self):
        for prior in ['rational', 'lab', 'tabpfnv2', '../lin', 'prior-999999', 'os.system', None, []]:
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                generate(priors=[prior], rows=8)

    def test_default_and_preset_order(self):
        self.assertEqual(GenerationConfig().priors, ['lin'])
        self.assertTrue(GenerationConfig().distinct_fill)
        self.assertEqual(DEFAULT8, ('mlp', 'tree', 'disc', 'lin', 'quad', 'gp', 'em', 'prod'))
        cfg, names = GenerationConfig(mode='dag', priors=['all24']).validated()
        self.assertEqual(set(names), EXPECTED)
        self.assertEqual(cfg.fct_types.split(',')[:8], list(DEFAULT8))
        _, default = GenerationConfig(mode='dag', priors=['default']).validated()
        _, explicit = GenerationConfig(mode='dag', priors=['default8']).validated()
        self.assertEqual(default, explicit)
        with self.assertRaises(ValueError):
            GenerationConfig(priors=['default8']).validated()

    def test_rng_and_reproducibility(self):
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        py = random.getstate(); ns = np.random.get_state(); ts = torch.random.get_rng_state()
        a = generate(rows=16, features=3, targets=2, seed=8)
        b = generate(rows=16, features=3, targets=2, seed=8)
        np.testing.assert_array_equal(a.data, b.data)
        self.assertEqual(a.data.shape, (16, 5))
        self.assertTrue(np.isfinite(a.data).all())
        self.assertEqual(random.getstate(), py)
        np.testing.assert_array_equal(np.random.get_state()[1], ns[1])
        self.assertEqual(np.random.get_state()[2:], ns[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), ts))
        self.assertFalse(np.array_equal(a.data, generate(rows=16, features=3, targets=2, seed=9).data))

    def test_dag_categories_and_hyperparameters(self):
        result = generate(mode='dag', rows=32, features=3, targets=2, seed=44,
                          priors=['lin', 'moe_linear_experts'], feature_categories=[0, 3, 0],
                          target_categories=[0, 2], graph_options={'min_n_nodes':4, 'max_n_nodes':4})
        self.assertEqual(result.data.shape, (32, 5))
        self.assertTrue(np.isin(result.data[:, 1], [0, 1, 2]).all())
        self.assertTrue(np.isin(result.data[:, 4], [0, 1]).all())
        direct = generate(rows=16, priors=['moe_linear_experts'], hyperparameters={'moelin_n_experts': 3})
        self.assertIn('moelin_n_experts', direct.metadata['hyperparameters_consumed'])
        with self.assertRaises(ValueError):
            generate(rows=8, hyperparameters={'unknown': 1})

    def test_all24_direct_smoke(self):
        for prior in sorted(EXPECTED):
            with self.subTest(prior=prior):
                result = generate(rows=16, features=3, targets=2, seed=13, priors=[prior])
                self.assertEqual(result.data.shape, (16, 5))
                self.assertTrue(np.isfinite(result.data).all())

    def test_distinct_fill_switch(self):
        from tablegen_core.priors import _distinct_fill as helper
        before = helper.ENABLED
        for enabled in [False, True]:
            result = generate(rows=16, targets=10, priors=['bourgouin_precipitation_type'], distinct_fill=enabled)
            self.assertEqual(result.metadata['distinct_fill']['enabled'], enabled)
            self.assertEqual(helper.ENABLED, before)

    def test_write_roundtrip_and_escape(self):
        result = generate(rows=8)
        result.columns[0] = '</script><script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'new'
            paths = result.write(folder)
            self.assertEqual(set(paths), {'table.csv', 'table.npy', 'metadata.json', 'report.html'})
            np.testing.assert_array_equal(np.load(folder/'table.npy', allow_pickle=False), result.data)
            loaded = np.loadtxt(folder/'table.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(loaded.astype(result.data.dtype), result.data)
            meta = json.loads((folder/'metadata.json').read_text())
            self.assertEqual(meta['shape'], [8, 5])
            self.assertNotIn('</script><script>alert(1)', (folder/'report.html').read_text())
            with self.assertRaises(FileExistsError):
                result.write(folder)

    def test_cli(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
        proc = subprocess.run([sys.executable, '-m', 'tablegen_core', 'list-priors', '--json'], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({e['name'] for e in json.loads(proc.stdout)}, EXPECTED)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'cli'
            proc = subprocess.run([sys.executable, '-m', 'tablegen_core', 'generate', '--rows', '8', '--prior', 'lin', '--seed', '7', '--output', str(folder)], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(np.load(folder/'table.npy').shape, (8, 5))
            bad = subprocess.run([sys.executable, '-m', 'tablegen_core', 'generate', '--prior', 'rational', '--output', str(Path(tmp)/'bad')], capture_output=True, text=True, env=env)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((Path(tmp)/'bad').exists())


if __name__ == '__main__':
    unittest.main()
