
import platform
import sys


def _try_import_torch() -> tuple[bool, str]:
    try:
        import torch

        info = [f"torch={torch.__version__}"]
        info.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info.append(f"cuda_device_count={torch.cuda.device_count()}")
            info.append(f"cuda_device_name={torch.cuda.get_device_name(0)}")
        if hasattr(torch.backends, "mps"):
            info.append(f"mps_available={torch.backends.mps.is_available()}")
        return True, ", ".join(info)
    except Exception as exc:
        return False, f"torch import failed: {exc}"


def _try_import_torchvision() -> tuple[bool, str]:
    try:
        import torchvision

        return True, f"torchvision={torchvision.__version__}"
    except Exception as exc:
        return False, f"torchvision import failed: {exc}"


def main() -> int:
    print("DL-Hub doctor")
    print(f"- python={sys.version.split()[0]}")
    print(f"- platform={platform.platform()}")

    torch_ok, torch_info = _try_import_torch()
    print(f"- {torch_info}")

    tv_ok, tv_info = _try_import_torchvision()
    print(f"- {tv_info}")

    if not torch_ok:
        print("\nNext step:")
        print("- Install PyTorch (platform-specific). See docs/INSTALL.md.")
        return 1

    if not tv_ok:
        print("\nNext step:")
        print("- Install torchvision for the vision track. See docs/INSTALL.md.")
        return 1

    print("\nLooks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
