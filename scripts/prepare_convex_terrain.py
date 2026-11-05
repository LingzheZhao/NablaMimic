"""Produce padded convex halfspaces from declared upstream CoACD hulls."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hulls', required=True)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--upstream-revision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.hulls)
    report = json.loads((source / 'result.json').read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    equations = []
    for index in range(len(report['hulls'])):
        mesh = trimesh.load(source / f'hull_{index}.ply', force='mesh')
        equations.append(ConvexHull(mesh.vertices).equations)
    width = max(len(item) for item in equations)
    normals = np.zeros((len(equations), width, 3), dtype=np.float32)
    offsets = np.zeros((len(equations), width), dtype=np.float32)
    valid = np.zeros((len(equations), width), dtype=bool)
    for index, item in enumerate(equations):
        normals[index, :len(item)] = item[:, :3]
        offsets[index, :len(item)] = item[:, 3]
        valid[index, :len(item)] = True
    np.savez(output / 'terrain.npz', normals=normals, offsets=offsets, valid_faces=valid)
    (output / 'contract.json').write_text(json.dumps(dict(
        format='nablamimic-convex-terrain-v1', artifact_id=args.artifact_id,
        source_directory=str(source), upstream_revision=args.upstream_revision,
        world_frame='upstream generated tile frame, metres; hull PLY already includes top alignment',
        inside_rule='normal dot world_position + offset <= 0',
        hulls=len(equations), padded_faces=width,
        geometry='convex hull of each supplied CoACD hull, without box substitution'), indent=2))


if __name__ == '__main__':
    main()
