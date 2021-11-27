from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch

import matplotlib.pyplot as plt


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))


config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
matching = Matching(config).eval().to(device)

input_dir = Path('assets/')
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path('dump_match_pairs/')
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))

timer = AverageTimer(newline=True)

name0 = 'scene0713_00_frame-001320.jpg'
name1 = 'scene0713_00_frame-002025.jpg'
stem0, stem1 = Path(name0).stem, Path(name1).stem

matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, 'png')
viz_eval_path = output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, 'png')

do_match = True
do_viz = True
resize_float = True

rot0 = 0
rot1 = 0

resize = [640]

# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, resize, rot0, resize_float)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, resize, rot1, resize_float)
if image0 is None or image1 is None:
    print('Problem reading image pair: {} {}'.format(
        input_dir / name0, input_dir / name1))

if do_match:
    # Perform the matching.q
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    timer.update('matcher')

    # Write the matches to disk.
    out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                   'matches': matches, 'match_confidence': conf}
    np.savez(str(matches_path), **out_matches)

valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

if do_viz:
    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    if rot0 != 0 or rot1 != 0:
        text.append('Rotation: {}:{}'.format(rot0, rot1))

    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(stem0, stem1),
    ]

    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, viz_path, True,
        False, False, 'Matches', small_text)

    timer.update('viz_match')

timer.print('Finished pair {:5} of {:5}'.format(1, 2))
