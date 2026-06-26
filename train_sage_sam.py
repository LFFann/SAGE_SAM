"""SAGE-SAM real-data training entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from Model.sage_model import DualSegmentor
from dataloader.builders import build_sage_dataloaders, resolve_dataset_root, validate_dataset_layout
from dataloader.calibration_split import load_or_create_calibration_manifest, compute_dataset_fingerprint
from dataloader.infinite_iterator import infinite_iterator
from engine.checkpoint import CheckpointManager, _safe_load
from engine.evaluator import evaluate_multiclass
from engine.logger import JSONLLogger
from engine.sage_trainer import SAGETrainer
from sage_ssl.structure_cache import StructureCacheReader
from sage_ssl.structure_calibration import StructureCalibration, calibrate_sam_structure
from utils.seed import seed_everything


def add_arg(parser, *names, **kwargs):
    parser.add_argument(*names, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAGE-SAM on a real semi-supervised dataset.")
    add_arg(parser, "--data_path", "--data-path", dest="data_path")
    add_arg(parser, "--dataset", default=None)
    add_arg(parser, "--image_size", "--image-size", dest="image_size", type=int, default=256)
    add_arg(parser, "--num_classes", "--num-classes", dest="num_classes", type=int, default=3)
    add_arg(parser, "--in_channels", "--in-channels", dest="in_channels", type=int, default=3)
    add_arg(parser, "--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    add_arg(parser, "--labeled_batch_size", "--labeled-batch-size", dest="labeled_batch_size", type=int, default=2)
    add_arg(parser, "--unlabeled_batch_size", "--unlabeled-batch-size", dest="unlabeled_batch_size", type=int, default=2)
    add_arg(parser, "--calibration_batch_size", "--calibration-batch-size", dest="calibration_batch_size", type=int, default=1)

    add_arg(parser, "--calibration_ratio", "--calibration-ratio", dest="calibration_ratio", type=float, default=0.2)
    add_arg(parser, "--calibration_manifest", "--calibration-manifest", dest="calibration_manifest")
    add_arg(parser, "--group_regex", "--group-regex", dest="group_regex")
    add_arg(parser, "--calibration_mode", "--calibration-mode", dest="calibration_mode", choices=["strict_once", "adaptive"], default="strict_once")
    add_arg(parser, "--calibration_interval", "--calibration-interval", dest="calibration_interval", type=int, default=500)
    add_arg(parser, "--conformal_alpha", "--conformal-alpha", dest="conformal_alpha", type=float, default=0.1)

    add_arg(parser, "--structure_cache", "--structure-cache", dest="structure_cache")
    add_arg(parser, "--structure_cache_mode", "--structure-cache-mode", dest="structure_cache_mode", choices=["required", "disabled"], default="required")
    add_arg(parser, "--structure_grid_size", "--structure-grid-size", dest="structure_grid_size", type=int, default=32)
    add_arg(parser, "--structure_calibration", "--structure-calibration", dest="structure_calibration")
    add_arg(parser, "--structure_target_precision", "--structure-target-precision", dest="structure_target_precision", type=float, default=0.9)
    add_arg(parser, "--ablation_supervised_only", "--ablation-supervised-only", dest="ablation_supervised_only", action="store_true")

    add_arg(parser, "--max_iterations", "--max-iterations", dest="max_iterations", type=int, default=10000)
    add_arg(parser, "--warmup_iterations", "--warmup-iterations", dest="warmup_iterations", type=int, default=1000)
    add_arg(parser, "--validation_interval", "--validation-interval", dest="validation_interval", type=int, default=250)
    add_arg(parser, "--checkpoint_interval", "--checkpoint-interval", dest="checkpoint_interval", type=int, default=500)
    add_arg(parser, "--log_interval", "--log-interval", dest="log_interval", type=int, default=20)
    add_arg(parser, "--unet_lr", "--unet-lr", dest="unet_lr", type=float, default=1e-3)
    add_arg(parser, "--vnet_lr", "--vnet-lr", dest="vnet_lr", type=float, default=1e-3)
    add_arg(parser, "--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-4)
    add_arg(parser, "--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    add_arg(parser, "--momentum", type=float, default=0.9)
    add_arg(parser, "--gradient_clip_norm", "--gradient-clip-norm", dest="gradient_clip_norm", type=float, default=0.0)
    add_arg(parser, "--amp", dest="amp", action="store_true", default=False)
    add_arg(parser, "--no_amp", "--no-amp", dest="amp", action="store_false")
    add_arg(parser, "--device", default=None)
    add_arg(parser, "--seed", type=int, default=2026)

    add_arg(parser, "--output_dir", "--output-dir", dest="output_dir", default="./outputs")
    add_arg(parser, "--experiment_name", "--experiment-name", dest="experiment_name", default="SAGE_SAM_3Class")
    add_arg(parser, "--resume")
    add_arg(parser, "--init_checkpoint", "--init-checkpoint", dest="init_checkpoint")
    add_arg(parser, "--save_final", "--save-final", dest="save_final", action="store_true", default=True)
    add_arg(parser, "--dry_run", "--dry-run", dest="dry_run", action="store_true")
    add_arg(parser, "--smoke", action="store_true")
    return parser


def resolve_output_dir(output_dir: str | Path, experiment_name: str) -> Path:
    return (Path(output_dir).expanduser() / experiment_name).resolve()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )


class TerminalProgress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.last_len = 0

    def _fit(self, message: str) -> str:
        width = shutil.get_terminal_size((100, 20)).columns
        width = max(min(width, 140), 60) - 1
        if len(message) <= width:
            return message
        return message[: max(width - 3, 1)] + "..."

    def update(self, message: str) -> None:
        message = self._fit(message)
        self.stream.write("\r\033[2K" + message)
        self.stream.flush()
        self.last_len = len(message)

    def write(self, message: str = "") -> None:
        if self.last_len:
            self.stream.write("\r\033[2K")
            self.last_len = 0
        if message:
            self.stream.write(message + "\n")
        else:
            self.stream.write("\n")
        self.stream.flush()


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def progress_bar(completed: int, total: int, width: int = 18) -> str:
    total = max(total, 1)
    filled = min(width, max(0, round(width * completed / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_metric(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_progress_status(
    record: dict,
    iteration: int,
    start_iteration: int,
    max_iterations: int,
    start_time: float,
    best_metric: float,
    last_metrics: dict | None = None,
    validating: bool = False,
) -> str:
    completed = iteration + 1
    total = max(max_iterations, 1)
    completed_since_start = iteration - start_iteration + 1
    elapsed = time.time() - start_time
    eta = (elapsed / max(completed_since_start, 1)) * max(max_iterations - iteration - 1, 0)
    percent = 100.0 * completed / total
    best_text = "n/a" if best_metric == float("-inf") else f"{best_metric:.4f}"
    val_text = "n/a"
    iou_text = "n/a"
    hd95_text = "n/a"
    if last_metrics:
        val_text = format_metric(last_metrics.get("fused_avg_dice"))
        iou_text = format_metric(last_metrics.get("fused_avg_iou"))
        hd95_text = format_metric(last_metrics.get("fused_avg_hd95"))
    stage = "val" if validating else ("warm" if record["stage"] == "warmup" else record["stage"])
    ssl_loss = float(record["loss_set"]) + 0.1 * float(record["loss_structure"])
    return (
        f"{progress_bar(completed, total)} {iteration + 1}/{max_iterations} {percent:4.1f}% "
        f"{stage} loss={record['loss_total']:.4f} "
        f"sup={record['loss_sup']:.4f} ssl={ssl_loss:.4f} "
        f"val={val_text} iou={iou_text} h={hd95_text} best={best_text} "
        f"eta={format_seconds(eta)}"
    )


def json_ready_args(args) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build_optimizer(args, model: DualSegmentor):
    groups = [
        {"params": model.UNet.parameters(), "lr": args.unet_lr, "name": "UNet"},
        {"params": model.VNet.parameters(), "lr": args.vnet_lr, "name": "VNet"},
    ]
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups, momentum=args.momentum, weight_decay=args.weight_decay)


def run_smoke(args) -> None:
    device = choose_device(args.device or "cpu")
    seed_everything(args.seed)
    model = DualSegmentor(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    optimizer = build_optimizer(args, model)
    trainer = SAGETrainer(model, optimizer, num_classes=args.num_classes)
    image = torch.rand(2, args.in_channels, 64, 64, device=device)
    label = torch.randint(0, args.num_classes, (2, 64, 64), device=device)
    output = trainer.step(labeled=(image, label))
    sources = trainer.optimizer_parameter_sources()
    assert sources["UNet"] + sources["VNet"] == model.trainable_parameter_report()["total"]
    print({"total_loss": float(output.total), "optimizer_sources": sources})


def prepare_run(args):
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init_checkpoint cannot be used together")
    if args.max_iterations <= 0:
        raise ValueError("--max_iterations must be > 0")
    if args.warmup_iterations > args.max_iterations:
        raise ValueError("--warmup_iterations cannot be greater than --max_iterations")
    if not args.data_path:
        raise ValueError("--data_path/--data-path is required outside --smoke")
    if not args.dataset:
        raise ValueError("--dataset is required outside --smoke")
    if args.structure_cache_mode == "disabled" and not args.ablation_supervised_only and args.warmup_iterations < args.max_iterations:
        raise ValueError("structure_cache_mode=disabled is only allowed for --ablation_supervised_only or smoke")

    dataset_root = resolve_dataset_root(args.data_path, args.dataset)
    validate_dataset_layout(dataset_root)
    run_dir = resolve_output_dir(args.output_dir, args.experiment_name)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and not args.dry_run:
        raise FileExistsError(f"Output directory already exists and is not empty: {run_dir}; use --resume or a new --experiment_name")
    (run_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    (run_dir / "args.json").write_text(json.dumps(json_ready_args(args), indent=2), encoding="utf-8")
    return dataset_root, run_dir


def load_init_checkpoint(path: str, model: DualSegmentor) -> dict:
    payload = _safe_load(path, map_location="cpu")
    state = payload.get("model", payload)
    current = model.state_dict()
    filtered = {}
    mismatched = []
    for key, value in state.items():
        if (key.startswith("UNet.") or key.startswith("VNet.")) and key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        elif key.startswith("UNet.") or key.startswith("VNet."):
            mismatched.append(key)
    report = model.load_state_dict(filtered, strict=False)
    return {"loaded_keys": len(filtered), "missing_keys": list(report.missing_keys), "unexpected_keys": list(report.unexpected_keys), "shape_mismatch": mismatched}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke:
        run_smoke(args)
        return

    device = choose_device(args.device)
    seed_everything(args.seed)
    dataset_root, run_dir = prepare_run(args)
    dataset_fingerprint = compute_dataset_fingerprint(dataset_root)
    manifest_path = Path(args.calibration_manifest) if args.calibration_manifest else run_dir / "calibration" / "calibration_manifest.json"
    manifest = load_or_create_calibration_manifest(
        dataset_root / "labeled" / "image",
        dataset_root / "labeled" / "mask",
        manifest_path,
        args.calibration_ratio,
        args.seed,
        args.num_classes,
        args.group_regex,
        dataset_fingerprint,
    )
    if manifest_path.resolve() != (run_dir / "calibration" / "calibration_manifest.json").resolve():
        (run_dir / "calibration" / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    loaders = build_sage_dataloaders(
        dataset_root=dataset_root,
        calibration_manifest=manifest,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        image_size=args.image_size,
        labeled_batch_size=args.labeled_batch_size,
        unlabeled_batch_size=args.unlabeled_batch_size,
        calibration_batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    structure_reader = None
    structure_info = {}
    structure_state = None
    if args.structure_cache_mode == "required":
        cache_root = Path(args.structure_cache) if args.structure_cache else Path(args.output_dir) / "structure_cache" / args.dataset
        required_ids = set(loaders["sample_ids"]["calibration"]) | set(loaders["sample_ids"]["unlabeled"])
        structure_reader = StructureCacheReader(cache_root, args.dataset, args.structure_grid_size, required_ids, dataset_fingerprint)
        structure_info = structure_reader.validate()
        structure_calibration_path = Path(args.structure_calibration) if args.structure_calibration else run_dir / "calibration" / "sam_structure_calibration.json"
        if structure_calibration_path.exists():
            structure_state = StructureCalibration.from_json(structure_calibration_path)
        else:
            structure_state = calibrate_sam_structure(
                loaders["calibration_loader"],
                structure_reader,
                args.num_classes,
                device,
                args.structure_target_precision,
                structure_calibration_path,
            )

    model = DualSegmentor(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    if args.init_checkpoint:
        logging.info("Loaded init checkpoint report: %s", load_init_checkpoint(args.init_checkpoint, model))
    optimizer = build_optimizer(args, model)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    trainer = SAGETrainer(
        model,
        optimizer,
        num_classes=args.num_classes,
        warmup_iterations=args.warmup_iterations,
        gradient_clip_norm=args.gradient_clip_norm,
        class_weights=loaders["class_weights"],
    )
    checkpoint_manager = CheckpointManager(run_dir / "checkpoints")
    start_iteration = 0
    best_metric = float("-inf")
    semantic_state = None
    if args.resume:
        payload = checkpoint_manager.load(args.resume, model, optimizer=optimizer, scaler=scaler, strict=False)
        if payload.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Resume checkpoint dataset_fingerprint does not match current dataset")
        start_iteration = int(payload["iteration"]) + 1
        best_metric = float(payload.get("best_metric", best_metric))
        manifest = payload.get("calibration_manifest", manifest)
        semantic_state = payload.get("semantic_calibration_state") or None
        trainer.semantic_calibration_state = semantic_state
        structure_state = payload.get("structure_calibration_state") or (structure_state.to_dict() if structure_state else None)

    if args.dry_run:
        supervised_batch = next(iter(loaders["supervised_loader"]))
        unlabeled_batch = next(iter(loaders["unlabeled_loader"]))
        calibration_batch = next(iter(loaders["calibration_loader"]))
        if structure_reader is not None:
            structure_reader.get(list(unlabeled_batch["sample_id"]))
        summary = {
            "dataset_root": str(dataset_root),
            "run_dir": str(run_dir),
            "supervised_batch": list(supervised_batch["image"].shape),
            "unlabeled_batch": list(unlabeled_batch["image"].shape),
            "calibration_batch": list(calibration_batch["image"].shape),
            "structure_cache": structure_info,
            "optimizer_sources": trainer.optimizer_parameter_sources(),
        }
        print(json.dumps(summary, indent=2, default=str))
        return

    train_logger = JSONLLogger(run_dir / "metrics" / "train.jsonl")
    progress = TerminalProgress()
    progress.write(f"Starting training. run_dir={run_dir}")
    supervised_iter = infinite_iterator(loaders["supervised_loader"])
    unlabeled_iter = infinite_iterator(loaders["unlabeled_loader"])
    train_start_time = time.time()
    last_metrics = None
    if args.warmup_iterations == 0 and semantic_state is None:
        semantic_state = trainer.calibrate_semantics(loaders["calibration_loader"], 0, args.conformal_alpha, device)
        (run_dir / "calibration" / "semantic_calibration.json").write_text(json.dumps(semantic_state, indent=2), encoding="utf-8")

    for iteration in range(start_iteration, args.max_iterations):
        t0 = time.time()
        labeled_batch = next(supervised_iter)
        unlabeled_batch = None
        structure_batch = None
        stage = "warmup"
        if iteration >= args.warmup_iterations and not args.ablation_supervised_only:
            if semantic_state is None or (args.calibration_mode == "adaptive" and iteration % args.calibration_interval == 0):
                semantic_state = trainer.calibrate_semantics(loaders["calibration_loader"], iteration, args.conformal_alpha, device)
                (run_dir / "calibration" / "semantic_calibration.json").write_text(json.dumps(semantic_state, indent=2), encoding="utf-8")
            unlabeled_batch = next(unlabeled_iter)
            if structure_reader is None:
                raise ValueError("SSL stage requires structure cache unless --ablation_supervised_only is set")
            structure_batch = structure_reader.get(list(unlabeled_batch["sample_id"]), device=device)
            stage = "ssl"

        output = trainer.train_step(labeled_batch, unlabeled_batch, structure_batch, iteration, device, scaler=scaler, amp_enabled=args.amp)
        record = {
            "iteration": iteration,
            "stage": stage,
            "loss_total": float(output.total),
            "loss_sup": float(output.supervised),
            "loss_set": float(output.set_loss),
            "loss_relation": float(output.relation_loss) if output.relation_loss is not None else 0.0,
            "loss_structure": float(output.structure_loss),
            "singleton_ratio": output.singleton_ratio,
            "ambiguous_ratio": output.ambiguous_ratio,
            "unknown_ratio": output.unknown_ratio,
            "mean_set_size": output.mean_set_size,
            "mean_hardness": float(output.hardness.float().mean()),
            "mean_instance_weight": output.mean_instance_weight,
            "learning_rate_unet": optimizer.param_groups[0]["lr"],
            "learning_rate_vnet": optimizer.param_groups[1]["lr"],
            "iteration_time": time.time() - t0,
            "gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0,
        }
        train_logger.log(**record)
        if args.log_interval > 0 and iteration % args.log_interval == 0:
            logging.info("iteration=%d stage=%s loss=%.6f", iteration, stage, record["loss_total"])
            progress.update(
                format_progress_status(
                    record,
                    iteration,
                    start_iteration,
                    args.max_iterations,
                    train_start_time,
                    best_metric,
                    last_metrics,
                )
            )

        def payload(metric_value: float):
            return checkpoint_manager.build_payload(
                iteration=iteration,
                best_metric=metric_value,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                dataset_root=str(dataset_root),
                dataset_fingerprint=dataset_fingerprint,
                calibration_manifest=manifest,
                semantic_calibration_state=trainer.semantic_calibration_state,
                structure_calibration_state=structure_state.to_dict() if hasattr(structure_state, "to_dict") else structure_state,
                class_weights=loaders["class_weights"],
                structure_cache_metadata=structure_info,
                git_commit=get_git_commit(),
            )

        if args.validation_interval > 0 and (iteration + 1) % args.validation_interval == 0:
            progress.update(
                format_progress_status(
                    record,
                    iteration,
                    start_iteration,
                    args.max_iterations,
                    train_start_time,
                    best_metric,
                    last_metrics,
                    validating=True,
                )
            )
            metrics = evaluate_multiclass(
                model,
                loaders["val_loader"],
                device,
                reliability_A=(trainer.semantic_calibration_state or {}).get("reliability_A"),
                reliability_B=(trainer.semantic_calibration_state or {}).get("reliability_B"),
                num_classes=args.num_classes,
                metrics_path=run_dir / "metrics" / "val.jsonl",
                iteration=iteration,
            )
            current = float(metrics["fused_avg_dice"])
            if current > best_metric:
                best_metric = current
                checkpoint_manager.save_best(payload(best_metric))
            logging.info("validation iteration=%d metrics=%s", iteration, json.dumps(metrics, sort_keys=True))
            last_metrics = metrics
            progress.update(
                format_progress_status(
                    record,
                    iteration,
                    start_iteration,
                    args.max_iterations,
                    train_start_time,
                    best_metric,
                    last_metrics,
                )
            )
        if args.checkpoint_interval > 0 and (iteration + 1) % args.checkpoint_interval == 0:
            checkpoint_manager.save_latest(payload(best_metric))

    if "record" in locals():
        progress.update(
            format_progress_status(
                record,
                args.max_iterations - 1,
                start_iteration,
                args.max_iterations,
                train_start_time,
                best_metric,
                last_metrics,
                validating=True,
            )
        )
    final_metrics = evaluate_multiclass(
        model,
        loaders["val_loader"],
        device,
        reliability_A=(trainer.semantic_calibration_state or {}).get("reliability_A"),
        reliability_B=(trainer.semantic_calibration_state or {}).get("reliability_B"),
        num_classes=args.num_classes,
        metrics_path=run_dir / "metrics" / "val.jsonl",
        iteration=args.max_iterations,
    )
    best_metric = max(best_metric, float(final_metrics["fused_avg_dice"]))
    logging.info("final_validation metrics=%s", json.dumps(final_metrics, sort_keys=True))
    last_metrics = final_metrics
    if "record" in locals():
        progress.update(
            format_progress_status(
                record,
                args.max_iterations - 1,
                start_iteration,
                args.max_iterations,
                train_start_time,
                best_metric,
                last_metrics,
            )
        )
    final_payload = checkpoint_manager.build_payload(
        iteration=args.max_iterations - 1,
        best_metric=best_metric,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        dataset_root=str(dataset_root),
        dataset_fingerprint=dataset_fingerprint,
        calibration_manifest=manifest,
        semantic_calibration_state=trainer.semantic_calibration_state,
        structure_calibration_state=structure_state.to_dict() if hasattr(structure_state, "to_dict") else structure_state,
        class_weights=loaders["class_weights"],
        structure_cache_metadata=structure_info,
        git_commit=get_git_commit(),
    )
    checkpoint_manager.save_latest(final_payload)
    if args.save_final:
        checkpoint_manager.save_final(final_payload)
    logging.info("Training complete. final fused_avg_dice=%.6f", final_metrics["fused_avg_dice"])
    progress.write(f"Training complete. final fused_avg_dice={final_metrics['fused_avg_dice']:.6f}")


if __name__ == "__main__":
    main()
