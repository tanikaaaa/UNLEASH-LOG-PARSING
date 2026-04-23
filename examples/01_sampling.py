# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import json
import os

from unleash.data.utils import load_loghub_dataset
from unleash.sampling.entropy_sampling import sampling as entropy_sampling
from unleash.sampling.lilac_sampling import sampling as lilac_sampling
from unleash.sampling.logppt_sampling import sampling as logppt_sampling

from config import benchmark
import argparse


# ✅ ALL 14 DATASETS
datasets = [
    "Apache","BGL","Hadoop","HDFS","HealthApp","HPC",
    "Linux","Mac","OpenSSH","OpenStack",
    "Proxifier","Spark","Thunderbird","Zookeeper"
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ✅ IMPORTANT FIX → default=None (so all datasets run)
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)

    parser.add_argument('--sampling_method', type=str, help='sampling method',
                        default='unleash', choices=['all', 'unleash', 'lilac', 'logppt'])

    args = parser.parse_args()

    # ✅ LOCAL DATA PATHS
    data_dir = "../data"
    output_dir = "../data"

    for dataset in datasets:

        # run only specific dataset if provided
        if args.dataset is not None and dataset != args.dataset:
            continue

        print(f"\n===== {dataset} =====")

        log_file = benchmark[dataset]['log_file']
        print(f"Loading {log_file}...")

        try:
            # ✅ Load RAW logs (TEXT mode)
            labelled_logs = load_loghub_dataset(
                dataset,
                data_dir,
                format="text",
                log_format=benchmark[dataset]["log_format"]
            )

        except Exception as e:
            print(f"❌ Skipping {dataset} (error loading data): {e}")
            continue

        # create sample folder
        os.makedirs(f'{output_dir}/{dataset}/samples', exist_ok=True)

        print(f"Loaded {len(labelled_logs)} logs.")

        # use 20% for validation
        k_rate = 0.2
        length = int(k_rate * len(labelled_logs))
        labelled_logs = labelled_logs[:length]

        # extract logs + labels
        raw_logs = labelled_logs['Content'].tolist()
        labels = labelled_logs['EventTemplate'].tolist()

        # save validation set
        with open(f'{output_dir}/{dataset}/validation.json', 'w') as f:
            for log, label in zip(raw_logs, labels):
                f.write(json.dumps({'log': log, 'template': label}) + '\n')

        # reduced shots (important for stability)
        shots = [8, 16, 32]

        # =========================
        # 🔹 ENTROPY SAMPLING
        # =========================
        if args.sampling_method in ['all', 'unleash']:
            print("Running Entropy Sampling...")
            sample_candidates = entropy_sampling(raw_logs, labels, shots)

            for shot, samples in sample_candidates.items():
                with open(f'{output_dir}/{dataset}/samples/unleash_{shot}.json', 'w') as f:
                    for sample in samples:
                        f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')

        # =========================
        # 🔹 LILAC SAMPLING
        # =========================
        if args.sampling_method in ['all', 'lilac']:
            print("Running LILAC Sampling...")
            sample_candidates = lilac_sampling(raw_logs, labels, shots)

            for shot, samples in sample_candidates.items():
                with open(f'{output_dir}/{dataset}/samples/lilac_{shot}.json', 'w') as f:
                    for sample in samples:
                        f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')

        # =========================
        # 🔹 LOGPPT SAMPLING
        # =========================
        if args.sampling_method in ['all', 'logppt']:
            print("Running LogPPT Sampling...")
            sample_candidates = logppt_sampling(raw_logs, labels, shots)

            for shot, samples in sample_candidates.items():
                with open(f'{output_dir}/{dataset}/samples/logppt_{shot}.json', 'w') as f:
                    for sample in samples:
                        f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')

        print(f"✅ Finished {dataset}")

    print("\n🎉 Sampling completed for all datasets.")