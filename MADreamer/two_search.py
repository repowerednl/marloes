import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num", type=int, default=10, help="Number of trials to run in search mode"
    )
    args = parser.parse_args()
    # run search.py for dreamer config
    subprocess.run(
        [
            "python",
            "search.py",
            "--search",
            "--analyze",
            "--no-parallel",
            "--cfg",
            "dreamer",
            "--num",
            str(args.num),
        ]
    )

    # run search.py for new_actor config
    subprocess.run(
        [
            "python",
            "search.py",
            "--search",
            "--analyze",
            "--no-parallel",
            "--cfg",
            "new_actor",
            "--num",
            str(args.num),
        ]
    )
