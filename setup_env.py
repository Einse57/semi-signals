"""Bootstrap script: installs requirements then uses MIM to install OpenMMLab packages."""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    # 1. Install base requirements
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # 2. Install OpenMMLab packages via MIM
    mim_packages = [
        "mmengine>=0.10.0",
        "mmcv>=2.1.0",
        "mmdet>=3.2.0",
        "mmpose>=1.3.0",
        "mmaction2>=1.2.0",
    ]
    for pkg in mim_packages:
        run([sys.executable, "-m", "mim", "install", pkg])

    print("\n=== Environment setup complete ===")


if __name__ == "__main__":
    main()
