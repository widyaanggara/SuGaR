"""Validate TogogLab commands against the checked-out native parsers without CUDA."""
import argparse
import ast
import importlib.util
from pathlib import Path
import unittest

from pipeline_tools.run_external_method import commands

ROOT = Path(__file__).resolve().parents[1]


def native_parser(repo, script):
    spec = importlib.util.spec_from_file_location('native_arguments', repo / 'arguments/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    if script not in ('metrics.py', 'metric.py'):
        module.ModelParams(parser)
        module.PipelineParams(parser)
        if script == 'train.py':
            module.OptimizationParams(parser)
    tree = ast.parse((repo / script).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'parser'
                and node.func.attr == 'add_argument'):
            continue
        args = [ast.literal_eval(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg == 'type' and isinstance(kw.value, ast.Name):
                kwargs[kw.arg] = {'str': str, 'int': int, 'float': float}[kw.value.id]
            else:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        parser.add_argument(*args, **kwargs)
    return parser


class NativeContractTests(unittest.TestCase):
    def test_every_generated_stage_is_accepted_by_native_parser(self):
        for method, folder in (('2dgs', '2DGS'), ('radegs', 'RaDe-GS')):
            repo = ROOT / 'external' / folder
            if not (repo / 'train.py').exists():
                self.skipTest('Clone the external repos to check native CLI contracts.')
            for stage, command in commands(method, 'python', Path('/dataset/scene'), Path('/output/scene'), 2):
                with self.subTest(method=method, stage=stage):
                    parser = native_parser(repo, command[2])
                    args = command[3:]
                    if stage == 'Training':
                        args += ['--start_checkpoint', '/output/chkpnt7000.pth']
                        if method == '2dgs':
                            args += ['--init_ply', '/anchors/initial.ply']
                    parsed = parser.parse_args(args)
                    self.assertIsNotNone(parsed)

    def test_scaffold_wrapper_matches_upstream_training_signature_and_parser(self):
        repo = ROOT / 'external/Scaffold-GS'
        if not (repo / 'train.py').exists():
            self.skipTest('Scaffold checkout unavailable.')
        parser = native_parser(repo, 'train.py')
        parser.parse_args(['-s', '/dataset', '-m', '/output', '--iterations', '30000', '-r', '1', '--eval'])
        tree = ast.parse((repo / 'train.py').read_text(encoding='utf-8'))
        training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training')
        self.assertEqual([a.arg for a in training.args.args[:9]],
            ['dataset', 'opt', 'pipe', 'dataset_name', 'testing_iterations', 'saving_iterations',
             'checkpoint_iterations', 'checkpoint', 'debug_from'])
        self.assertIn('logger', [a.arg for a in training.args.args])


if __name__ == '__main__':
    unittest.main()
