"""Inspect HDF5 groups, attributes and array metadata without loading arrays."""

import argparse
from pathlib import Path

import h5py


def show_attributes(obj, indent):
    for name in obj.attrs:
        value = repr(obj.attrs[name])
        if len(value) > 240:
            value = value[:240] + "..."
        print(f"{indent}@{name}: {value}")


def inspect_file(path):
    print(f"\nFile: {path}\n/")
    with h5py.File(path, "r") as handle:
        show_attributes(handle, "  ")

        def visit(name, obj):
            indent = "  " * (name.count("/") + 1)
            label = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{label}: shape={obj.shape}, dtype={obj.dtype}, compression={obj.compression}")
            else:
                print(f"{indent}{label}/")
            show_attributes(obj, indent + "  ")

        handle.visititems(visit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="HDF5 file or directory to search recursively")
    parser.add_argument("--limit", type=int, default=1, help="Files to inspect (default: 1; 0: all)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")
    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(p for p in args.path.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"})
    print(f"Found {len(files)} HDF5 file(s).")
    if not files:
        parser.error("No HDF5 files found")
    failures = 0
    for path in files[:args.limit] if args.limit else files:
        try:
            inspect_file(path)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {path}: {exc}")
            failures += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
