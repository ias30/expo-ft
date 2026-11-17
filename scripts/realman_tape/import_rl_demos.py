#!/usr/bin/env python3
"""Convert labelled RealMan LeRobot episodes to the RL replay NPZ format."""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/openpi_rtc/src'),
               str(ROOT/'third_party/openpi_rtc/packages/openpi-client/src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--root', required=True, help='Local LeRobot dataset directory')
    p.add_argument('--labels', required=True, help='JSON object: episode index -> true/false success')
    p.add_argument('--output', required=True)
    p.add_argument('--prompt', default='pick up the tape from the rotating disc')
    args = p.parse_args()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from expo_ft.realman.demos import labelled_episodes
    labels = json.loads(pathlib.Path(args.labels).read_text())
    if not labels or any(type(v) is not bool for v in labels.values()):
        raise ValueError('Each label must be an explicit JSON boolean; omitted episodes are not imported')
    dataset = LeRobotDataset(args.repo_id, root=pathlib.Path(args.root), episodes=[int(k) for k in labels])
    count = 0
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for index, ep in labelled_episodes(dataset, labels, args.prompt):
        path = output/f'demo_{index:06d}.npz'
        if path.exists():
            raise FileExistsError(path)
        ep.save(path)
        count += 1
    print(f'Imported {count} labelled episodes. Final row supplies terminal observation; its unobserved action is omitted.')


if __name__ == '__main__':
    main()
