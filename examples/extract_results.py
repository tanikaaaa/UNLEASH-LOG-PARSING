import os
import subprocess
import re

datasets = [
    "Apache","BGL","Hadoop","HDFS","HealthApp","HPC",
    "Linux","Mac","OpenSSH","OpenStack",
    "Proxifier","Spark","Thunderbird","Zookeeper"
]

results = []

for d in datasets:
    print(f"Running {d}...")
    
    cmd = f"python 03_evaluation.py --output_dir None --dataset {d}"
    output = subprocess.getoutput(cmd)

    ga = re.search(r'Grouping_Accuracy \(GA\): ([0-9.]+)', output)
    pa = re.search(r'Parsing_Accuracy \(PA\): ([0-9.]+)', output)

    ga = ga.group(1) if ga else "NA"
    pa = pa.group(1) if pa else "NA"

    results.append(f"{d}, {ga}, {pa}")

# save clean CSV
with open("final_results.csv", "w") as f:
    f.write("Dataset,GA,PA\n")
    for r in results:
        f.write(r + "\n")

print("✅ Results saved to final_results.csv")