#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import csv
import time
import sys
import concurrent.futures
import pandas as pd
import itertools

# ============================================================================
# Configuration
# ============================================================================

source_dir = "../test/aa.data"
hmm_file = "../data/combined.hmm"
result_dir = "../test/HMM"
log_file = "../test/HMM/HMM_log.txt"
summary_file = "../test/HMM/HMM_result.csv"
annotation_tsv = "../data/vfam.annotations.tsv"
final_output = "../test/HMM/combined_hmmscan_results_with_func.csv"

# Virus-Host combination parameters
host_file = "../train_model/dataset.csv"
virus_host_output = "../test/virus_host.csv"

max_workers = 32

# ============================================================================
# Part 1: HMMSCAN Processing
# ============================================================================

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

log_handle = open(log_file, "w")
log_handle.write("=== HMMSCAN PROCESSING STARTED AT {0} ===\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
log_handle.write("Source directory: {0}\nHMM file: {1}\nResult directory: {2}\n\n".format(source_dir, hmm_file, result_dir))

fasta_files = [f for f in os.listdir(source_dir) if f.endswith(('.fasta', '.fa', '.faa'))]
log_handle.write("Found {0} FASTA files to process\n\n".format(len(fasta_files)))
log_handle.flush()

def run_hmmscan(filename):
    fasta_path = os.path.join(source_dir, filename)
    result_path = os.path.join(result_dir, "{0}.tbl".format(filename))
    
    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        return (filename, 0, "Skipped (already exists)")
    
    cmd = ["hmmscan", "-E", "1e-5", "--tblout", result_path, hmm_file, fasta_path]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        return (filename, process.returncode, stderr.decode('utf-8'))
    except Exception as e:
        return (filename, -1, str(e))

print("Running HMMSCAN on {0} files...".format(len(fasta_files)))

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_hmmscan, f): f for f in fasta_files}
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        filename = futures[future]
        try:
            fname, returncode, stderr = future.result()
            log_handle.write("[{0}/{1}] Finished: {2} (Return code: {3})\n".format(
                i+1, len(fasta_files), fname, returncode))
            if returncode != 0:
                log_handle.write("  STDERR: {0}\n".format(stderr))
        except Exception as e:
            log_handle.write("  ERROR processing {0}: {1}\n".format(filename, e))
        log_handle.flush()

log_handle.write("\n=== ALL HMMSCAN TASKS COMPLETED AT {0} ===\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
log_handle.write("Now creating summary CSV file...\n")
log_handle.flush()

# ============================================================================
# Part 2: Create Summary CSV
# ============================================================================

print("Creating summary CSV file...")

columns = ["filename", "target_name", "query_name"]

# Python 2/3 compatible file opening
if sys.version_info[0] >= 3:
    csvfile = open(summary_file, 'w', newline='')
else:
    csvfile = open(summary_file, 'wb')

try:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    
    for filename in os.listdir(result_dir):
        if not filename.endswith('.tbl'):
            continue
        original_filename = filename[:-4]  
        result_file_path = os.path.join(result_dir, filename)

        seen_query_names = set()

        with open(result_file_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                target_name = parts[0]
                query_name = parts[2]

                if query_name in seen_query_names:
                    continue

                seen_query_names.add(query_name)
                writer.writerow([original_filename, target_name, query_name])
finally:
    csvfile.close()

log_handle.write("Summary CSV file created: {0}\n".format(summary_file))
log_handle.flush()

# ============================================================================
# Part 3: Annotation Processing
# ============================================================================

print("Processing annotations and creating final output...")

usecols = ["filename", "target_name", "query_name"]
df = pd.read_csv(summary_file, dtype=str, usecols=usecols)
df['filename'] = df['filename'].str.replace(r'\.fasta$', '', regex=True)
combined_df = df.drop_duplicates()

annotation_df = pd.read_csv(
    annotation_tsv,
    sep='\t',
    dtype=str
)
annotation_df = annotation_df[['#GroupName', 'FunctionalCategory', 'ConsensusFunctionalDescription']]

merged_df = pd.merge(
    combined_df,
    annotation_df,
    left_on='target_name',
    right_on='#GroupName',
    how='left'
)

func_to_category = {
    "XhXpXs": "VOG_Xs",
    "XhXp": "VOG_XhXp",
    "XhXpXrXs": "VOG_XrXs",
    "XhXrXs": "VOG_XrXs",
    "Xs": "VOG_Xs",
    "Xu": "VOG_Non",
    "XpXrXs": "VOG_XrXs",
    "XhXs": "VOG_Xs",
    "XhXpXr": "VOG_Xr",
    "XrXs": "VOG_XrXs",
    "Xr": "VOG_Xr",
    "XhXr": "VOG_Xr",
    "XpXs": "VOG_Xs",
    "Xh": "VOG_XhXp",
    "XpXr": "VOG_Xr",
    "Xp": "VOG_XhXp"
}

merged_df['virus_category'] = merged_df['FunctionalCategory'].map(func_to_category)
merged_df['virus_category'] = merged_df['virus_category'].fillna('Other')

merged_df.to_csv(final_output, index=False)

log_handle.write("Final annotated CSV file created: {0}\n".format(final_output))
log_handle.flush()

print("\n=== OUTPUT 1 COMPLETE ===")
print("File 1: {0}".format(final_output))
print("Total records: {0}".format(len(merged_df)))

# ============================================================================
# Part 4: Create Virus-Host Combinations
# ============================================================================

print("\nCreating virus-host combinations...")

virus_list = merged_df['filename'].unique()

host_df = pd.read_csv(host_file)
host_list = host_df['host_filename'].unique()

combinations = list(itertools.product(virus_list, host_list))
comb_df = pd.DataFrame(combinations, columns=['SeqName.virus', 'SeqName.host'])

comb_df.to_csv(virus_host_output, index=False)

log_handle.write("Virus-Host combination file created: {0}\n".format(virus_host_output))
log_handle.write("Process completed successfully at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
log_handle.close()

