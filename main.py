import sys
from OffManifoldLearning import training, make_dataset

def main():
    if len(sys.argv) < 2:
        raise ValueError("Provide subcommand (e.g. training)")

    cmd = sys.argv[1]

    if cmd == "training":
        training.training()
    elif cmd == "distances_baseline":
        make_dataset.calc_dist('baseline')
    elif cmd == "distances_post_rehab":
        make_dataset.calc_dist('post_rehab')
    elif cmd == "var_expl_baseline":
        make_dataset.calc_var_expl('baseline')
    elif cmd == "var_expl_post_rehab":
        make_dataset.calc_var_expl('post_rehab')
    elif cmd == "pool_log_training":
        training.pool_log_training('controller_training')
    elif cmd == "calc_A_diff":
        training.calc_A_diff()
    elif cmd == "rehabilitation":
        training.rehabilitation()
    elif cmd == "make_dataset_baseline":
        make_dataset.make_dataset_baseline()
    elif cmd == "make_dataset_postrehab":
        make_dataset.make_dataset_postrehab()
    else:
        raise ValueError(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()