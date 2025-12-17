## Libraries & Imports:
from pathlib import Path
from scripts.utils import load_csv
import subprocess

def run_script(script_name):
    """Run a script via subprocess (assumes Python module style: python -m scripts.script_name)"""
    print(f"\n=== Running {script_name} ===")
    subprocess.run(["python", "-m", f"scripts.{script_name}"], check=True)
    print(f"=== Completed {script_name} ===\n")

def main():
    # Step 0: Preprocess articles
    run_script("1_data_parsing")

    # Step 1: Context Extraction with Coreference Resolution
    run_script("2_coref_context_extraction")

    # Step 2: Description Extraction
    run_script("3_description_extraction")

    # Step 3: Description Clustering
    run_script("4_description_clustering")

    # Step 4: Manual Cluster Evaluation
    print("Manual Cluster Evaluation: Step 4")
    print("Please inspect the clusters in processed/clusters_victims.csv and clusters_shooters.csv.")
    print("Refine if necessary before proceeding to cross-outlet analysis.")
    input("Press Enter when ready to continue...")

    # Step 5: Cross-Outlet Frequency Analysis
    run_script("5_cross_outlet_analysis")

    # Step 6: Statistical Hypothesis Testing
    run_script("6_hypothesis_testing")

    # Step 7: Visualization
    run_script("visualization") 

if __name__ == "__main__":
    main()
