import os
import subprocess
import pandas as pd
import tempfile

# === CONFIG ===
hmm_folder = "profiles"
targets = [
    "WithMCogs_tagged.fasta",
    "No_MCogs_with_Cogs_tagged.fasta",
    "unidentified.fasta",
    "combined.fasta"
]
results_folder = "hmmsearch_combined"
evalue_cutoff = 1e-5
threads = 4

os.makedirs(results_folder, exist_ok=True)

# === FUNCTIONS ===
def run_hmmsearch(hmm_file, target_file, tmp_tbl):
    """Run hmmsearch on one target and write to temporary table."""
    cmd = [
        "hmmsearch",
        "--cpu", str(threads),
        "-E", str(evalue_cutoff),
        "--tblout", tmp_tbl,
        hmm_file,
        target_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def parse_tblout(tbl_file, hmm_name, fasta_name):
    """Parse hmmsearch --tblout output into a list of dicts."""
    hits = []
    with open(tbl_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 18:
                hits.append({
                    "query_profile": hmm_name,
                    "target_seq": parts[0],
                    "accession": parts[1],
                    "evalue": float(parts[4]),
                    "score": float(parts[5]),
                    "bias": float(parts[6]),
                    "target_file": os.path.basename(fasta_name)
                })
    return hits

# === MAIN ===
for hmm in os.listdir(hmm_folder):
    if not hmm.endswith(".hmm"):
        continue

    hmm_path = os.path.join(hmm_folder, hmm)
    base = os.path.splitext(hmm)[0]
    all_hits = []

    print(f"\n🔍 Processing profile: {base}")

    for target in targets:
        print(f"   ↳ Searching against {os.path.basename(target)} ...")
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_tbl = tmp.name
        run_hmmsearch(hmm_path, target, tmp_tbl)
        hits = parse_tblout(tmp_tbl, base, target)
        all_hits.extend(hits)
        os.remove(tmp_tbl)

    # Convert to DataFrame and save combined table
    if all_hits:
        df = pd.DataFrame(all_hits)
        df.sort_values(by="score", ascending=False, inplace=True)
        out_path = os.path.join(results_folder, f"{base}_combined_hits.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved combined hits: {out_path}")
    else:
        print(f"⚠️ No hits found for {base}")

print("\n🎯 All profiles processed.")

