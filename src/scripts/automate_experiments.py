#!/usr/bin/env python3
"""
Automate training and evaluation experiments.

Training combinations:
- Generators: biggan, adm, glide
- Compressions: raw, jpeg96, jpeg95, jpeg90
- All 12 combinations trained with --train-percent 0.1

For each trained model, evaluate on all combinations of:
- Test generators: biggan, adm, glide
- Test compressions: raw, jpeg90, jpeg95, jpeg96
- 12 evaluations per training model
"""

import subprocess
import sys
from pathlib import Path

# Training configurations
TRAIN_COMPRESSIONS = ['raw', 'jpeg96', 'jpeg95', 'jpeg90']
TRAIN_GENERATORS = ['biggan', 'adm', 'glide']

# Test configurations
TEST_COMPRESSIONS = ['raw', 'jpeg90', 'jpeg95', 'jpeg96']
TEST_GENERATORS = ['biggan', 'adm', 'glide']

def run_command(cmd: str, description: str) -> None:
    """Run a command with progress output."""
    print(f"\n{'='*50}")
    print(f"STARTING: {description}")
    print(f"Command: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"FINISHED: {description}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {description}: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        sys.exit(1)

def main() -> None:
    total_trainings = len(TRAIN_COMPRESSIONS) * len(TRAIN_GENERATORS)
    total_evals_per_training = len(TEST_COMPRESSIONS) * len(TEST_GENERATORS)
    total_evals = total_trainings * total_evals_per_training
    
    print(f"Total trainings: {total_trainings}")
    print(f"Total evaluations: {total_evals}")
    print("Starting automation...")
    
    training_count = 0
    eval_count = 0
    
    for train_comp in TRAIN_COMPRESSIONS:
        for train_gen in TRAIN_GENERATORS:
            training_count += 1
            desc = f"Training {training_count}/{total_trainings}: gen={train_gen}, comp={train_comp}"
            
            cmd = (
                f"python -m src.scripts.train_detector "
                f"--config configs/default.yaml "
                f"--train-generators {train_gen} "
                f"--train-compression {train_comp} "
                f"--train-percent 0.1 "
                f"--model resnet50"
            )
            run_command(cmd, desc)
            
            # Checkpoint path
            checkpoint = f"outputs/models/best_resnet50_train-{train_gen}_comp-{train_comp}_pct-0.1.pt"
            
            # Ensure checkpoint exists
            if not Path(checkpoint).exists():
                print(f"WARNING: Checkpoint {checkpoint} not found, skipping evaluations")
                continue
            
            # Evaluations for this training
            for test_comp in TEST_COMPRESSIONS:
                for test_gen in TEST_GENERATORS:
                    eval_count += 1
                    eval_desc = (
                        f"Eval {eval_count}/{total_evals}: "
                        f"train_gen={train_gen}, train_comp={train_comp}, "
                        f"test_gen={test_gen}, test_comp={test_comp}"
                    )
                    
                    cmd = (
                        f"python -m src.scripts.evaluate_detector "
                        f"--config configs/default.yaml "
                        f"--checkpoint {checkpoint} "
                        f"--eval-generators {test_gen} "
                        f"--test-compression {test_comp}"
                    )
                    run_command(cmd, eval_desc)
    
    print(f"\n{'='*50}")
    print("AUTOMATION COMPLETE")
    print(f"Completed {training_count} trainings and {eval_count} evaluations")
    print('='*50)

if __name__ == '__main__':
    main()