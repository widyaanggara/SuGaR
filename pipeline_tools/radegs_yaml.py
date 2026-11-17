"""Read the native RaDe-GS YAML using its own environment and validators."""
import argparse
import json
import sys
from pathlib import Path


def build_plan(repo, config_path, scene, output):
    sys.path.insert(0, str(repo))
    from research_pipeline.config import load_config, repository_parser
    from research_pipeline.commands import commands
    config = load_config(Path(config_path))
    # Routing and geometry evaluation belong to TogogLab, not the donor machine.
    config['dataset']['root'] = str(scene)
    config['output']['root'] = str(output)
    config['geometry_evaluation']['enabled'] = False
    config['scaffold_initialization']['enabled'] = False
    parsed = repository_parser('training').parse_args(config['training']['extra_args'])
    if parsed.iterations != 30000 or not parsed.eval:
        raise ValueError('Dashboard requires 30000 iterations and --eval for checkpoint/metric compatibility.')
    if not all(config[key]['enabled'] for key in ('rendering', 'image_metrics', 'mesh')):
        raise ValueError('YAML must enable rendering, image_metrics and mesh.')
    stages = commands(config, Path(scene), Path(output))
    stages['training'].extend(['--checkpoint_iterations', '7000', '15000', '30000'])
    return {'config': config, 'commands': stages, 'resolution': parsed.resolution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--guard', type=Path)
    args = parser.parse_args()
    if args.guard:
        sys.path.insert(0, str(args.repo))
        import logging
        from research_pipeline.resource_guard import wait_until_ready
        logging.basicConfig(level=logging.INFO)
        ready, error = wait_until_ready(json.loads(args.guard.read_text(encoding='utf-8')))
        if not ready:
            raise RuntimeError(error)
    else:
        print(json.dumps(build_plan(args.repo, args.config, args.scene, args.output)))


if __name__ == '__main__':
    main()
