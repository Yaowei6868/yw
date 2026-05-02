import importlib
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    print("== Environment Check ==")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())

    import torch

    print("torch:", version_of("torch"))
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda.device_count:", torch.cuda.device_count())
        print("cuda.device_name:", torch.cuda.get_device_name(0))

    import torch_geometric
    import numpy
    import pandas
    import sklearn
    import networkx
    import omegaconf

    print("torch_geometric:", version_of("torch_geometric"))
    print("numpy:", version_of("numpy"))
    print("pandas:", version_of("pandas"))
    print("scikit-learn:", version_of("sklearn"))
    print("networkx:", version_of("networkx"))
    print("omegaconf:", version_of("omegaconf"))

    from fraud_detection import Trainer  # noqa: F401

    print("fraud_detection import: OK")
    print("== Check Passed ==")


if __name__ == "__main__":
    main()
