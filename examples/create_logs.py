import pandas as pd
import os

datasets = [
    "Apache","BGL","Hadoop","HDFS","HealthApp","HPC",
    "Linux","Mac","OpenSSH","OpenStack",
    "Proxifier","Spark","Thunderbird","Zookeeper"
]

for dataset in datasets:
    try:
        # ✅ correct CSV location (your setup)
        csv_path = f"../../UNLEASH_log_parser/data/{dataset}/{dataset}_2k.log_structured_corrected.csv"

        # ✅ define output_dir properly (THIS WAS MISSING)
        output_dir = f"../data/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/{dataset}_full.log"

        df = pd.read_csv(csv_path)

        if "Content" not in df.columns:
            print(f"❌ {dataset}: No Content column")
            continue

        df["Content"].to_csv(output_path, index=False, header=False)

        print(f"✅ Created {dataset}_full.log")

    except Exception as e:
        print(f"❌ Skipped {dataset}: {e}")