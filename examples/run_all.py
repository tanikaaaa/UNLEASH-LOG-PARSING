import os

datasets = [
    "Apache","BGL","Hadoop","HDFS","HealthApp","HPC",
    "Linux","Mac","OpenSSH","OpenStack",
    "Proxifier","Spark","Thunderbird","Zookeeper"
]

for dataset in datasets:
    print(f"\n===== Running {dataset} =====")

    command = (
        f"python 02_run_unleash.py "
        f"--train_file ../data/{dataset}/samples/unleash_32.json "
        f"--dataset_name {dataset} "
        f"--log_file \"../data/{dataset}/{dataset}_full.log\" "
        f"--output_dir None"
    )

    print("Running command:", command)
    os.system(command)