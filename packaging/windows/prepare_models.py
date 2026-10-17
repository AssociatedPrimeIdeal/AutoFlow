"""Export trusted local nnUNet checkpoints as a portable inference-only bundle.

Run with autoflow311 on a server. Training inputs, optimizer state, training
case maps, logs, and patient identifiers are not needed by nnUNet inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch


DATASET_KEYS = {"channel_names", "modality", "labels", "file_ending",
                "overwrite_image_reader_writer", "regions_class_order"}
PLANS_KEYS = {"dataset_name", "plans_name", "original_median_spacing_after_transp",
              "original_median_shape_after_transp", "image_reader_writer",
              "transpose_forward", "transpose_backward", "configurations",
              "experiment_planner_used", "label_manager",
              "foreground_intensity_properties_per_channel"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(source, target, profile):
    source, target = Path(source).resolve(), Path(target)
    target.mkdir(parents=True)
    for filename, keys in (("dataset.json", DATASET_KEYS), ("plans.json", PLANS_KEYS)):
        value = json.loads((source / filename).read_text(encoding="utf-8"))
        value = {key: item for key, item in value.items() if key in keys}
        (target / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    checkpoints = []
    for fold in sorted(source.glob("fold_*")):
        for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
            original = fold / name
            if not original.is_file():
                continue
            # Inputs are explicitly supplied trusted training checkpoints.
            loaded = torch.load(original, map_location="cpu", weights_only=False)
            inference = {
                "network_weights": loaded["network_weights"],
                "trainer_name": loaded["trainer_name"],
                "init_args": {"configuration": loaded["init_args"]["configuration"]},
                "inference_allowed_mirroring_axes": loaded.get("inference_allowed_mirroring_axes"),
            }
            if inference["trainer_name"] != "nnUNetTrainer":
                raise ValueError(f"{profile}: custom inference trainer must be packaged explicitly")
            destination = target / fold.name / name
            destination.parent.mkdir(exist_ok=True)
            torch.save(inference, destination)
            verified = torch.load(destination, map_location="cpu", weights_only=False)
            if loaded["network_weights"].keys() != verified["network_weights"].keys():
                raise ValueError(f"{profile}: weight keys changed")
            if not all(torch.equal(tensor, verified["network_weights"][key])
                       for key, tensor in loaded["network_weights"].items()):
                raise ValueError(f"{profile}: weights changed")
            checkpoints.append({"file": f"{fold.name}/{name}", "sha256": sha256(destination),
                                "source_sha256": sha256(original), "weights_verified_equal": True})
            del loaded, inference, verified
    if not checkpoints or not (target / "fold_all" / "checkpoint_final.pth").is_file():
        raise ValueError(f"{profile}: fold_all/checkpoint_final.pth is required")
    return {"profile": profile, "checkpoints": checkpoints,
            "dataset_sha256": sha256(target / "dataset.json"),
            "plans_sha256": sha256(target / "plans.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-3d", required=True)
    parser.add_argument("--model-4d", required=True)
    parser.add_argument("--model-bundled", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autoflow_models_") as temp:
        models = Path(temp) / "models"
        manifest = {"schema_version": 1, "inference_only": True, "profiles": []}
        for profile, source in (("nnunet3d", args.model_3d), ("nnunet4d", args.model_4d),
                                ("nnunet3d-bundled", args.model_bundled)):
            manifest["profiles"].append(export_model(source, models / profile, profile))
            print(f"Verified unchanged inference weights: {profile}", flush=True)
        (models / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(models.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(models.parent).as_posix())
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(args.output), "sha256": digest, "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
