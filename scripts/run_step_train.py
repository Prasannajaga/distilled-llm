from __future__ import annotations

import subprocess
import sys
from importlib import resources


def main() -> None:
    step_train_path = resources.files("scripts").joinpath("step-train.sh")
    cmd = ["bash", str(step_train_path), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
