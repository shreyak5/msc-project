import argparse
import csv
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Merge results.csv/skipped_videos.txt/summary.json from N shard_i/ subfolders '
                    '(produced by run_evaluation_dataset.py --num_shards N) into one combined result.')
    parser.add_argument('--output_dir', type=str, required=True,
                         help='Directory containing shard_0/, shard_1/, ... subfolders; merged output is written here')
    parser.add_argument('--num_shards', type=int, required=True)
    args = parser.parse_args()

    merged_rows = []
    fieldnames = None
    skipped_lines = []
    totals = {'num_videos_total': 0, 'num_videos_processed': 0, 'num_videos_skipped': 0}
    shard_elapsed_seconds = []

    for i in range(args.num_shards):
        shard_dir = os.path.join(args.output_dir, f'shard_{i}')

        with open(os.path.join(shard_dir, 'results.csv'), newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            merged_rows.extend(reader)

        skipped_path = os.path.join(shard_dir, 'skipped_videos.txt')
        if os.path.exists(skipped_path):
            with open(skipped_path) as f:
                skipped_lines.extend(f.readlines())

        with open(os.path.join(shard_dir, 'summary.json')) as f:
            shard_summary = json.load(f)
        for key in totals:
            totals[key] += shard_summary[key]
        shard_elapsed_seconds.append(shard_summary['elapsed_seconds'])

    results_path = os.path.join(args.output_dir, 'results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    skipped_path = os.path.join(args.output_dir, 'skipped_videos.txt')
    with open(skipped_path, 'w') as f:
        f.writelines(skipped_lines)

    # First mean (per-video, across that video's frames) was already computed by
    # run_evaluation_dataset.py and stored as each row's "<name>_mean". Here we only
    # take the second mean (across all videos' first-means, pooled across every shard) -
    # never a third mean over shard-level summaries.
    metric_names = sorted({name.rsplit('_mean', 1)[0] for name in fieldnames if name.endswith('_mean')})
    overall_mean = {}
    overall_std = {}
    for name in metric_names:
        means = [float(row[f'{name}_mean']) for row in merged_rows if row[f'{name}_mean'] != '']
        overall_mean[name] = float(np.mean(means)) if means else None
        overall_std[name] = float(np.std(means)) if means else None

    summary = {
        **totals,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'max_shard_elapsed_seconds': max(shard_elapsed_seconds),
        'total_compute_seconds': sum(shard_elapsed_seconds),
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Merged {args.num_shards} shards. Results saved to {results_path}, summary saved to {summary_path}')


if __name__ == '__main__':
    main()