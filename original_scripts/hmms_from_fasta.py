import os
import subprocess

# === CONFIG ===
input_folder = "COGs"
alignment_folder = "alignments"
hmm_folder = "profiles"

# Choose your alignment tool: "mafft" or "clustalo"
aligner = "clustalo"  # or "mafft"   (FIX: the comment used to say "clustalo" twice)

# === SETUP ===
os.makedirs(alignment_folder, exist_ok=True)
os.makedirs(hmm_folder, exist_ok=True)

# === FUNCTIONS ===
def run_alignment(input_fasta, output_alignment):
    """Align sequences using MAFFT or Clustal Omega."""
    if aligner == "mafft":
        cmd = ["mafft", "--auto", input_fasta]
    elif aligner == "clustalo":
        cmd = ["clustalo", "-i", input_fasta, "-o", output_alignment, "--force"]
    else:
        raise ValueError("Unknown aligner")

    try:
        with open(output_alignment, "w") as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠️  Alignment failed for {input_fasta}, skipping...")
        return False

def run_hmmbuild(alignment_file, output_hmm):
    """Build HMM profile using hmmbuild (HMMER)."""
    cmd = ["hmmbuild", output_hmm, alignment_file]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠️  HMM build failed for {alignment_file}, skipping...")
        return False

def create_single_sequence_hmm(fasta_file, hmm_output):
    """
    Create a pseudo-HMM from a single sequence using hmmbuild.
    This creates a profile that essentially matches the single sequence.
    """
    try:
        # First create a temporary alignment with just the single sequence
        temp_aln = hmm_output.replace('.hmm', '.aln')
        
        with open(fasta_file, 'r') as fin, open(temp_aln, 'w') as fout:
            for line in fin:
                fout.write(line)
        
        # Build HMM from the single sequence "alignment"
        cmd = ["hmmbuild", hmm_output, temp_aln]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up temporary alignment
        os.remove(temp_aln)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create single-sequence HMM: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating single-sequence HMM: {e}")
        return False



# === MAIN LOOP ===
successful = 0
total = 0
singles_count = 0

for file in os.listdir(input_folder):
    if file.endswith(".fasta") or file.endswith(".fa"):
        total += 1
        fasta_path = os.path.join(input_folder, file)
        base_name = os.path.splitext(file)[0]

        # Check if FASTA has sequences
        with open(fasta_path, 'r') as f:
            seq_count = sum(1 for line in f if line.startswith('>'))
        
        aln_path = os.path.join(alignment_folder, base_name + ".aln")
        hmm_path = os.path.join(hmm_folder, base_name + ".hmm")

        if seq_count == 1:
            # Single sequence - create HMM directly
            print(f"🧬 Single sequence {file}, creating direct HMM...")
            if create_single_sequence_hmm(fasta_path, hmm_path):
                singles_count += 1
                print(f"✅ Single-sequence HMM: {base_name}.hmm")
            else:
                print(f"❌ Single-sequence HMM failed: {base_name}")
                
        elif seq_count < 2:
            print(f"⚠️  Skipping {file}: only {seq_count} sequences (need at least 1)")
            continue
        else:
            # Multiple sequences - normal alignment and HMM building
            print(f"🧩 Processing {file} ({seq_count} sequences)...")
            
            if run_alignment(fasta_path, aln_path):
                if run_hmmbuild(aln_path, hmm_path):
                    successful += 1
                    print(f"✅ Done: {base_name}.hmm")
                else:
                    print(f"❌ HMM build failed: {base_name}")
            else:
                print(f"❌ Alignment failed: {base_name}")

print(f"🎯 Profiles built: {successful} multi-sequence + {singles_count} single-sequence = {successful + singles_count}/{total} successful")

