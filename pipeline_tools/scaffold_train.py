"""Run official Scaffold training without its GPU auto-selection or post-training evaluation.

The upstream source is loaded in memory; the external checkout is never modified.
Use a separate Python environment with the official Scaffold-GS dependencies.
"""
import argparse
import ast
from pathlib import Path
import sys


def training_module(source):
    tree = ast.parse(source)
    torch_line = next(node.lineno for node in tree.body
                      if isinstance(node, ast.Import) and any(x.name == 'torch' for x in node.names))
    # Upstream executes nvidia-smi/grep and overwrites CUDA_VISIBLE_DEVICES before
    # importing torch. Keep imports, but omit that platform-specific preamble.
    tree.body = [node for node in tree.body if node.lineno >= torch_line or isinstance(node, (ast.Import, ast.ImportFrom))]
    return tree


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', required=True)
    options, remaining = pre.parse_known_args()
    repo = Path(options.repo).resolve()
    sys.path.insert(0, str(repo))
    path = repo / 'train.py'
    namespace = {'__name__': 'scaffold_training', '__file__': str(path)}
    exec(compile(training_module(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    parser = argparse.ArgumentParser()
    model = namespace['ModelParams'](parser)
    optimization = namespace['OptimizationParams'](parser)
    pipeline = namespace['PipelineParams'](parser)
    args = parser.parse_args(remaining)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    logger = namespace['get_logger'](args.model_path)
    namespace['safe_state'](False)
    namespace['network_gui'].try_connect = lambda: None
    namespace['training'](model.extract(args), optimization.extract(args), pipeline.extract(args),
                          Path(args.source_path).name, [], [args.iterations], [], None, -1,
                          wandb=None, logger=logger)


if __name__ == '__main__':
    main()
