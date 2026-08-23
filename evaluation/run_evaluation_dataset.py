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


def load_prior_progress(results_path, skipped_path, keys):
    """Reconstructs resume state from a previous (possibly incomplete) run's output, so a
    shard that got killed partway through (SLURM timeout, transient failure) can resume
    without redoing already-completed clips. Slow per-clip methods (e.g. pixel3dmm, ~8min/
    clip) make this the difference between losing minutes vs. losing a day of GPU-hours.

    Returns (done_names, per_set_means, num_skipped): done_names covers both successfully
    processed clips (from results_path) and previously-skipped ones (from skipped_path, so
    they aren't retried indefinitely); per_set_means is seeded from results_path's rows so
    the eventual summary.json still reflects every completed clip, not just ones processed
    in this particular run.
    """
    done_names = set()
    per_set_means = {name: [] for name in keys}
    num_skipped = 0

    if os.path.exists(results_path):
        with open(results_path, newline='') as f:
            for row in csv.DictReader(f):
                done_names.add(row['name'])
                for name_ in keys:
                    mean = row.get(f'{name_}_mean')
                    if mean:
                        per_set_means[name_].append(float(mean))

    if os.path.exists(skipped_path):
        with open(skipped_path) as f:
            for line in f:
                name = line.split('\t', 1)[0].strip()
                if name:
                    done_names.add(name)
                    num_skipped += 1

    return done_names, per_set_means, num_skipped


def load_prior_elapsed_seconds(summary_path):
    if not os.path.exists(summary_path):
        return 0.0
    with open(summary_path) as f:
        return json.load(f).get('elapsed_seconds', 0.0)


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
    parser.add_argument('--dataset_name', type=str, default='inference',
                         help="Dataset name for the visibility-score cache bucket (see eval_core.VISIBILITY_CACHE_ROOT). "
                              "Pass an indexed dataset's real name (e.g. csl_daily, how2sign, phoenix2014t) to land in "
                              "- and reuse - the same cache entries training's own data loading already populated for "
                              "that dataset; left at the default 'inference' bucket for ad-hoc/non-indexed inputs.")
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Optional trained checkpoint for the selected method; omit to sanity-test '
                              'the untrained model (ignored by --method smirk, which uses its own fixed checkpoint)')
    parser.add_argument('--kernel_radius', type=int, default=4,
                         help='--method ours_kernel_smooth only: kernel-smoothing window radius')
    parser.add_argument('--kernel_sigma', type=float, default=2.0,
                         help='--method ours_kernel_smooth only: Gaussian kernel std (frames)')
    parser.add_argument('--kernel_temperature', type=float, default=0.1,
                         help='--method ours_kernel_smooth only: visibility softmax temperature')
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

    # ours_kernel_smooth-only kwargs, forwarded into its setup() - every other
    # method's setup() has a fixed signature (not **kwargs), so this must stay
    # conditional or it would break them with an unexpected-argument error.
    method_kwargs = None
    if args.method == 'ours_kernel_smooth':
        method_kwargs = {
            'radius': args.kernel_radius, 'sigma': args.kernel_sigma,
            'temperature': args.kernel_temperature, 'dataset_name': args.dataset_name,
        }

    evaluators = build_evaluators(args.method, args.device, args.crop_size,
                                   crop_scale=args.crop_scale, checkpoint_path=args.checkpoint,
                                   dataset_name=args.dataset_name, method_kwargs=method_kwargs)

    all_clips = list_clips(args.input_dir, args.image_seq)
    clips = all_clips[args.shard_index::args.num_shards]
    total = len(clips)
    print(f'Found {len(all_clips)} clips in {args.input_dir}, shard {args.shard_index}/{args.num_shards} handles {total}')

    results_path = os.path.join(output_dir, f'{args.method}_results.csv')
    skipped_path = os.path.join(output_dir, f'{args.method}_skipped_videos.txt')
    summary_path = os.path.join(output_dir, f'{args.method}_summary.json')

    fieldnames = ['name'] + [f'{name}_mean' for name in keys] + [f'{name}_std' for name in keys] + \
        [f'{name}_valid_frames' for name in keys] + [f'{name}_total_frames' for name in keys]

    resuming = os.path.exists(results_path)
    already_done, per_set_means, num_skipped = load_prior_progress(results_path, skipped_path, keys)
    prior_elapsed_seconds = load_prior_elapsed_seconds(summary_path)

    remaining_clips = [c for c in clips if os.path.basename(c.rstrip('/')) not in already_done]
    if resuming:
        print(f'Resuming: {len(already_done)}/{total} clips already done, '
              f'{len(remaining_clips)} remaining')

    start_time = time.time()

    with open(results_path, 'a' if resuming else 'w', newline='') as results_file, \
            open(skipped_path, 'a' if resuming else 'w') as skipped_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        if not resuming:
            writer.writeheader()
            results_file.flush()

        for i, clip_path in enumerate(remaining_clips, start=len(already_done) + 1):
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

    elapsed_seconds = prior_elapsed_seconds + (time.time() - start_time)

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
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Skipped {num_skipped}/{total} videos — see {skipped_path}')
    print(f'Results saved to {results_path}, summary saved to {summary_path}')


if __name__ == '__main__':
    main()
