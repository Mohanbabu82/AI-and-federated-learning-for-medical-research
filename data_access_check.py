"""
Step 1: Verify MedMNIST dataset access on a CPU-only, 8 GB RAM laptop.

Checks 5 specialty subsets (= 5 simulated hospitals):
    PathMNIST  (pathology)
    DermaMNIST (dermatology)
    BloodMNIST (blood)
    OCTMNIST   (retina)
    TissueMNIST(kidney tissue)

For each: downloads (cached after first run), reports dataset stats,
and confirms a single sample can be converted to 3-channel RGB and
resized to 28x28. Does NOT load full datasets into memory - only a
small sample is pulled per subset to stay within the RAM budget.

This script only verifies data access. No model/training code here.
"""

import sys
import traceback

import numpy as np
from PIL import Image

SPECIALTIES = {
    "PathMNIST": "pathmnist",
    "DermaMNIST": "dermamnist",
    "BloodMNIST": "bloodmnist",
    "OCTMNIST": "octmnist",
    "TissueMNIST": "tissuemnist",
}

SAMPLE_CHECK_N = 5  # only pull a handful of images per subset for the check


def to_rgb_28(img):
    """Convert a PIL image (grayscale or RGB) to 3-channel RGB, resized to 28x28."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.BILINEAR)
    return img


def check_specialty(display_name, flag_name):
    try:
        import medmnist
        from medmnist import INFO

        if flag_name not in INFO:
            print(f"[{display_name}] FAILED: '{flag_name}' not found in medmnist.INFO")
            return None

        info = INFO[flag_name]
        DataClass = getattr(medmnist, info["python_class"])

        # size=28 keeps images small (matches local_debug image_size=28)
        dataset = DataClass(split="train", download=True, size=28)

        n_classes = len(info["label"])
        n_train = len(dataset)

        # Pull only a small sample (lazy __getitem__) rather than loading all images.
        sample_imgs = []
        sample_shapes = set()
        pixel_min, pixel_max = None, None
        for i in range(min(SAMPLE_CHECK_N, n_train)):
            img, label = dataset[i]
            arr = np.array(img)
            sample_shapes.add(arr.shape)
            lo, hi = arr.min(), arr.max()
            pixel_min = lo if pixel_min is None else min(pixel_min, lo)
            pixel_max = hi if pixel_max is None else max(pixel_max, hi)
            sample_imgs.append(img)

        raw_shape = sample_imgs[0].size + (
            (1,) if sample_imgs[0].mode == "L" else (3,)
        )

        # Confirm grayscale -> RGB -> 28x28 conversion works and is consistent
        converted = [to_rgb_28(im) for im in sample_imgs]
        converted_shapes = {np.array(c).shape for c in converted}
        consistent = len(converted_shapes) == 1 and next(iter(converted_shapes)) == (28, 28, 3)

        return {
            "name": display_name,
            "flag": flag_name,
            "n_classes": n_classes,
            "n_train": n_train,
            "raw_shape": raw_shape,
            "pixel_min": pixel_min,
            "pixel_max": pixel_max,
            "converted_shape": next(iter(converted_shapes)) if converted_shapes else None,
            "consistent": consistent,
            "error": None,
        }

    except Exception as e:
        return {
            "name": display_name,
            "flag": flag_name,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main():
    print("=" * 90)
    print("MedMNIST Data Access Check (CPU-only, 8GB RAM laptop)")
    print("=" * 90)

    try:
        import torch
        import torchvision
        print(f"torch:       {torch.__version__} (cuda available: {torch.cuda.is_available()})")
        print(f"torchvision: {torchvision.__version__}")
        import medmnist
        print(f"medmnist:    {medmnist.__version__}")
    except Exception as e:
        print(f"FATAL: could not import core libraries: {e}")
        sys.exit(1)

    print("-" * 90)

    results = []
    for display_name, flag_name in SPECIALTIES.items():
        print(f"Checking {display_name} ({flag_name}) ...")
        res = check_specialty(display_name, flag_name)
        results.append(res)
        if res.get("error"):
            print(f"  -> ERROR: {res['error']}")
        else:
            print(f"  -> OK ({res['n_train']} train images, {res['n_classes']} classes)")

    print()
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Specialty':<12} {'#Classes':>8} {'#Train':>8} {'ImgShape':>12} {'PixMin':>7} {'PixMax':>7}  Status"
    print(header)
    print("-" * len(header))

    any_failure = False
    for res in results:
        if res.get("error"):
            any_failure = True
            print(f"{res['name']:<12} {'--':>8} {'--':>8} {'--':>12} {'--':>7} {'--':>7}  FAILED: {res['error']}")
        else:
            shape_str = "x".join(str(d) for d in res["raw_shape"])
            print(
                f"{res['name']:<12} {res['n_classes']:>8} {res['n_train']:>8} "
                f"{shape_str:>12} {res['pixel_min']:>7} {res['pixel_max']:>7}  OK"
            )

    print()
    print("=" * 90)
    print("RGB CONVERSION + RESIZE CONSISTENCY (grayscale/RGB -> 3-channel RGB, 28x28)")
    print("=" * 90)
    all_consistent = True
    for res in results:
        if res.get("error"):
            continue
        status = "OK" if res["consistent"] else "MISMATCH"
        if not res["consistent"]:
            all_consistent = False
        print(f"{res['name']:<12} converted_shape={res['converted_shape']}  {status}")

    print()
    print("=" * 90)
    if any_failure:
        print("RESULT: One or more specialties FAILED to load. See errors above "
              "(likely network/download issue). Do not proceed until resolved.")
        sys.exit(1)
    elif not all_consistent:
        print("RESULT: All specialties downloaded, but RGB/resize conversion was "
              "INCONSISTENT across subsets.")
        sys.exit(1)
    else:
        print("RESULT: SUCCESS. All 5 specialty subsets accessible, cached, and "
              "convertible to a consistent 28x28x3 format.")
        sys.exit(0)


if __name__ == "__main__":
    main()
