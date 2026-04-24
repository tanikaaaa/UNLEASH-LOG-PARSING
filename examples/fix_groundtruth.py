import os
import shutil

datasets = [
    "Apache","BGL","Hadoop","HDFS","HealthApp","HPC",
    "Linux","Mac","OpenSSH","OpenStack",
    "Proxifier","Spark","Thunderbird","Zookeeper"
]

for d in datasets:
    src = f"../../UNLEASH_log_parser/data/{d}/{d}_2k.log_structured_corrected.csv"
    dst_dir = f"../datasets/loghub-2.0/{d}"
    dst = f"{dst_dir}/{d}_full.log_structured.csv"

    os.makedirs(dst_dir, exist_ok=True)

    try:
        shutil.copy(src, dst)
        print(f"✅ Copied {d}")
    except Exception as e:
        print(f"❌ {d}: {e}")