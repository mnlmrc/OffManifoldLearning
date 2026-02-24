import sys
from OffManifoldLearning import training, make_dataset

def main():
    if len(sys.argv) < 2:
        raise ValueError("Provide subcommand (e.g. training)")

    cmd = sys.argv[1]
    args = sys.argv[2:]  # everything after subcommand

    if cmd == "training":
        training.main(args)
    elif cmd == "make_dataset":
        make_dataset.make_dataset_baseline()
    elif cmd == "make_dataset_postrehab":
        make_dataset.make_dataset_postrehab()
    else:
        raise ValueError(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()