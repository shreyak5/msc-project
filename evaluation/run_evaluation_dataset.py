import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from eval_core import build_evaluators, evaluate_clip, result_keys, METHOD_REGISTRY
from metrics import summarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import timestamped_out_dir  # noqa: E402

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def list_clips(input_dir, image_seq):
    if image_seq:
        entries = sorted(
            os.path.join(input_dir, name) for name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, name)))
    else:
        entries = sorted(
            os.path.join(input_dir, name) for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
            and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS)
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a 3D reconstruction method against FAN/MediaPipe 2D landmarks across a whole dataset directory.')
    parser.add_argument('--input_dir', type=str, required=True,
                         help='Directory of video files, or directory of image-sequence subdirectories')
    parser.add_argument('--image_seq', action='store_true',
                         help='Treat each immediate subdirectory of input_dir as one image-sequence clip')
    parser.add_argument('--fps', type=int, default=30, help='FPS applied to every clip when --image_seq')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run detectors/method on')
    parser.add_argument('--crop_scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser.add_argument('--method', type=str, default='smirk', choices=sorted(METHOD_REGISTRY.keys()))
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Optional trained checkpoint for the selected method; omit to sanity-test '
                              'the untrained model (ignored by --method smirk, which uses its own fixed checkpoint)')
    parser.add_argument('--output_dir', type=str, default='evaluation/output_dataset',
                         help='Directory to save results.csv, skipped_videos.txt and summary.json')
    parser.add_argument('--num_shards', type=int, default=1,
                         help='Split the clip list into this many shards (e.g. to run one shard per GPU in parallel)')
    parser.add_argument('--shard_index', type=int, default=0,
                         help='Which shard this process handles, in [0, num_shards)')
    args = parser.parse_args()

    keys = result_keys()

    if args.num_shards > 1:
        # No timestamped_out_dir here: shard_index processes are launched in parallel
        # by the caller, and merge_dataset_shards.py needs every shard to agree on the
        # same output_dir/shard_i/ path deterministically (not by luck of landing in the
        # same clock-minute).
        output_dir = os.path.join(args.output_dir, f'shard_{args.shard_index}')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = timestamped_out_dir(args.output_dir)

    evaluators = build_evaluators(args.method, args.device, args.crop_size,
                                   crop_scale=args.crop_scale, checkpoint_path=args.checkpoint)

    all_clips = list_clips(args.input_dir, args.image_seq)
    clips = all_clips[args.shard_index::args.num_shards]
    total = len(clips)
    print(f'Found {len(all_clips)} clips in {args.input_dir}, shard {args.shard_index}/{args.num_shards} handles {total}')

    results_path = os.path.join(output_dir, f'{args.method}_results.csv')
    skipped_path = os.path.join(output_dir, f'{args.method}_skipped_videos.txt')

    fieldnames = ['name'] + [f'{name}_mean' for name in keys] + [f'{name}_std' for name in keys] + \
        [f'{name}_valid_frames' for name in keys] + [f'{name}_total_frames' for name in keys]

    per_set_means = {name: [] for name in keys}
    num_skipped = 0

    start_time = time.time()

    with open(results_path, 'w', newline='') as results_file, open(skipped_path, 'w') as skipped_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        results_file.flush()

        for i, clip_path in enumerate(clips, start=1):
            name = os.path.basename(clip_path.rstrip('/'))

            try:
                frames, _video_fps, _input_name = load_frames(clip_path, args.image_seq, args.fps)
                if not frames:
                    raise ValueError('no frames found')
                errors = evaluate_clip(frames, args.crop_scale, args.crop_size, evaluators, name)
                clip_summary = {name_: summarize(errors[name_]) for name_ in keys}
            except Exception as e:
                num_skipped += 1
                print(f'[{i}/{total}] SKIPPED {name}: {e}')
                skipped_file.write(f'{name}\t{e}\n')
                skipped_file.flush()
                continue

            if all(clip_summary[name_]['num_valid_frames'] == 0 for name_ in keys):
                num_skipped += 1
                print(f'[{i}/{total}] SKIPPED {name}: no valid frames detected')
                skipped_file.write(f'{name}\tno valid frames detected\n')
                skipped_file.flush()
                continue

            row = {'name': name}
            progress_parts = []
            for name_ in keys:
                mean = clip_summary[name_]['mean']
                std = clip_summary[name_]['std']
                valid = clip_summary[name_]['num_valid_frames']
                total_frames = clip_summary[name_]['num_frames']
                row[f'{name_}_mean'] = mean
                row[f'{name_}_std'] = std
                row[f'{name_}_valid_frames'] = valid
                row[f'{name_}_total_frames'] = total_frames
                if mean is not None:
                    per_set_means[name_].append(mean)
                    progress_parts.append(f'{name_}={mean:.2f} ({valid}/{total_frames})')
                else:
                    progress_parts.append(f'{name_}=n/a ({valid}/{total_frames})')

            writer.writerow(row)
            results_file.flush()
            print(f'[{i}/{total}] {name}: ' + ' '.join(progress_parts))

    elapsed_seconds = time.time() - start_time

    overall_mean = {
        name_: (float(np.mean(per_set_means[name_])) if per_set_means[name_] else None)
        for name_ in keys
    }
    overall_std = {
        name_: (float(np.std(per_set_means[name_])) if per_set_means[name_] else None)
        for name_ in keys
    }
    summary = {
        'num_videos_total': total,
        'num_videos_processed': total - num_skipped,
        'num_videos_skipped': num_skipped,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'elapsed_seconds': elapsed_seconds,
    }
    summary_path = os.path.join(output_dir, f'{args.method}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Skipped {num_skipped}/{total} videos — see {skipped_path}')
    print(f'Results saved to {results_path}, summary saved to {summary_path}')


if __name__ == '__main__':
    main()
