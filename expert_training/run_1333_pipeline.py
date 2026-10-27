#!/usr/bin/env python3
"""Run the documented non-generative experiment with durable stage markers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("00_splits", "00_prepare_splits.py", "00_splits/manifest.json"),
    ("01_calibration", "01_build_calibration.py", "01_calibration/manifest.json"),
    ("02_partition", "02_select_adaptive_blocks.py", "02_partition/partition.json"),
    ("03_full_groups", "03_build_full_block_groups.py", "03_full_groups/manifest.json"),
    ("04_assign", "04_assign_sample_groups.py", "04_sample_groups/manifest.json"),
    ("05_experts", "05_train_experts.py", "05_experts/run_meta.json"),
    ("06_routers", "06_train_block_routers.py", "06_routers/run_meta.json"),
    ("07_evaluation", "07_evaluate.py", "07_evaluation/metrics.json"),
    ("08_check", "08_check_artifacts.py", "07_evaluation/metrics.json"),
]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text())


def environment(run_dir, gpus):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_DATASETS_CACHE"] = str(run_dir / "cache" / "datasets")
    env["HF_HOME"] = str(run_dir / "cache" / "huggingface")
    env["TMPDIR"] = str(run_dir / "cache" / "tmp")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    packages = Path(sysconfig.get_paths()["purelib"])
    libraries = sorted((packages / "nvidia").glob("*/lib")) + [packages / "torch" / "lib"]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in libraries if path.is_dir()) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    nccl = packages / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    if nccl.exists():
        env["LD_PRELOAD"] = str(nccl) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    return env


def check_config(config, nproc):
    groups = config.get("num_groups")
    if isinstance(groups, dict):
        if not groups or sorted(int(key) for key in groups) != list(range(len(groups))):
            raise ValueError("num_groups must define consecutive block IDs starting at zero")
        if any(int(value) < 1 for value in groups.values()):
            raise ValueError("Each block needs at least one expert")
    elif not isinstance(groups, int) or groups < 1:
        raise ValueError("num_groups must be a positive integer or a per-block mapping")
    positive_fields = (
        "max_length", "calibration_num_groups", "calibration_batch_size",
        "full_signature_batch_size", "probe_train_batch_size",
        "expert_train_batch_size", "expert_effective_batch_size", "evaluation_batch_size",
    )
    for key in positive_fields:
        if int(config.get(key, 0)) < 1:
            raise ValueError(f"{key} must be positive")
    signature_world = int(config.get("signature_world_size", 1))
    if signature_world not in (1, nproc):
        raise ValueError("signature_world_size must be one or match the launcher GPU count")
    if signature_world > 1:
        for key in ("calibration_batch_size", "calibration_validation_batch_size", "full_signature_batch_size"):
            if int(config.get(key, config["calibration_batch_size"])) % signature_world:
                raise ValueError(f"{key} must be divisible by signature_world_size")
    train_batch = int(config["expert_train_batch_size"])
    effective_batch = int(config["expert_effective_batch_size"])
    if effective_batch % (nproc * train_batch):
        raise ValueError("Expert effective batch must be divisible by GPU count times local train batch")
    for key in ("expert_validation_batch_size", "router_train_batch_size", "router_validation_batch_size", "calibration_validation_batch_size"):
        if key in config and int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if config.get("pipeline_mode") == "joint_backbone":
        if groups != {"0": 1, "1": 2, "2": 3, "3": 3} or int(config["max_length"]) not in (1024, 2048):
            raise ValueError("The new backbone requires K=1233 and max_length 1024 or 2048")
        if (config["min_blocks"], config["max_blocks"], config["block_min_length"], config["block_max_length"]) != (4, 4, 10, 10):
            raise ValueError("The new backbone uses four fixed 10-layer blocks")
        joint = load_json(Path(config["joint_training_config"]))
        if joint["effective_batch_size"] != nproc * joint["local_batch_size"] or joint["effective_batch_size"] not in (8, 16):
            raise ValueError("Joint training must realize global batch 8 or 16 without accumulation")
        if joint["eval_local_batch_size"] != 8 or joint.get("evaluate_test_after_training", True):
            raise ValueError("Backbone validation uses batch 8; final test follows KDA/RAG selection")
        if (nproc * train_batch, effective_batch) != (joint["effective_batch_size"], joint["effective_batch_size"]):
            raise ValueError("Expert and joint training must use the same global batch without accumulation")
        if (config.get("expert_validation_batch_size"), config["evaluation_batch_size"], config.get("oracle_candidate_batch_size")) != (8, 8, 8):
            raise ValueError("The new backbone requires validation and test batch 8")
        if config.get("expert_gradient_accumulation_steps", 1) != 1:
            raise ValueError("The new backbone requires accumulation 1")
        source = load_json(Path(joint["source_config"]))
        if source != config or Path(joint["run_dir"]).resolve() != Path(config["run_dir"]).resolve() / "06_joint":
            raise ValueError("Joint training must use this run's source configuration and output directory")
    for key in ("model_name_or_path", "source_train_csv", "source_test_csv", "id2label_csv"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_1333_bs16.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    stages = list(STAGES)
    if config.get("pipeline_mode") == "joint_backbone":
        stages = stages[:6] + [("06_joint", "20_train_joint_moe.py", "06_joint/progress.json")]
    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) != len(set(gpu_ids)) or not all(item.isdigit() for item in gpu_ids):
        raise ValueError("GPU IDs must be distinct integers")
    check_config(config, len(gpu_ids))
    run_dir = Path(config["run_dir"]).resolve()
    if not run_dir.is_relative_to(ROOT / "runs") or run_dir == ROOT / "runs":
        raise ValueError("This runner only writes an individual run under new_layerwise/runs")
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (logs / "launcher.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--gpus", args.gpus],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(f"Submitted launcher PID={child.pid}; progress={run_dir / 'progress.json'}", flush=True)
        return

    with (run_dir / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = environment(run_dir, args.gpus)
        state_dir = run_dir / "pipeline_state"
        state_dir.mkdir(exist_ok=True)
        snapshot = state_dir / "config.json"
        if snapshot.exists() and load_json(snapshot) != config:
            raise ValueError("Run configuration changed; use a new run_dir instead of mixing artifacts")
        write_json(snapshot, config)
        if config.get("pipeline_mode") == "joint_backbone":
            joint_snapshot = state_dir / "joint_config.json"
            joint_config = load_json(Path(config["joint_training_config"]))
            if joint_snapshot.exists() and load_json(joint_snapshot) != joint_config:
                raise ValueError("Joint configuration changed; use a new run_dir")
            write_json(joint_snapshot, joint_config)
        source_paths = [ROOT / script for script in ["run_1333_pipeline.py"]]
        source_paths += [ROOT / "scripts" / stage[1] for stage in stages]
        source_paths += [ROOT / "adaptive_blockwise" / f"{name}.py" for name in ("io_utils", "signals", "formulas", "modeling", "experiment_contract")]
        if config.get("pipeline_mode") == "joint_backbone":
            source_paths.append(ROOT / "adaptive_blockwise/joint_moe.py")
        source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
        source_snapshot = state_dir / "source_hashes.json"
        if source_snapshot.exists() and load_json(source_snapshot) != source_hashes:
            raise ValueError("Experiment source changed since launch; review before resuming this run")
        write_json(source_snapshot, source_hashes)
        progress_path = run_dir / "progress.json"
        for index, (name, script, artifact) in enumerate(stages):
            done = state_dir / f"{name}.done.json"
            if done.exists():
                if not (run_dir / artifact).exists():
                    raise FileNotFoundError(f"Completed stage artifact is missing: {artifact}")
                print(f"Already complete: {name}", flush=True)
                continue
            if config.get("pipeline_mode") == "joint_backbone" and name in ("05_experts", "06_joint"):
                smoke_path = run_dir / ("preflight_smoke_4gpu.json" if len(gpu_ids) > 1 else "preflight_smoke.json")
                smoke = load_json(smoke_path)
                if (not smoke.get("passed") or smoke.get("config") != config
                        or smoke.get("world_size", 1) != len(gpu_ids)
                        or (len(gpu_ids) > 1 and smoke.get("joint_config") != joint_config)):
                    raise ValueError("Complete the matching K=1233 DDP training smoke with this configuration first")
            stage_config = Path(config["joint_training_config"]) if name == "06_joint" else config_path
            command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(stage_config)]
            if index in (0, 4):
                command.append("--overwrite")
            if (index == 5 or name == "06_joint"
                    or (name in ("01_calibration", "03_full_groups", "04_assign") and int(config.get("signature_world_size", 1)) > 1)):
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpu_ids)}"] + command[1:]
            stage_env = dict(env)
            if index in (0, 2, 8):
                stage_env["CUDA_VISIBLE_DEVICES"] = ""
            log_path = logs / f"{name}.log"
            progress = {"status": "running", "stage": name, "launcher_pid": os.getpid(), "log": str(log_path), "started_at": time.time()}
            write_json(progress_path, progress)
            print(f"Starting {name}: {command}", flush=True)
            with log_path.open("a", buffering=1) as log:
                log.write(f"\nSTART {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n")
                child = subprocess.Popen(command, cwd=ROOT, env=stage_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                progress["child_pid"] = child.pid
                write_json(progress_path, progress)
                try:
                    for line in child.stdout:
                        log.write(line)
                        progress.update(last_output=line.rstrip()[-1200:], updated_at=time.time())
                        write_json(progress_path, progress)
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            if code or not (run_dir / artifact).exists():
                progress.update(status="failed", exit_code=code, finished_at=time.time())
                write_json(progress_path, progress)
                raise RuntimeError(f"Stage {name} failed; inspect {log_path}. Later stages were not started.")
            if name == "06_joint" and load_json(run_dir / artifact).get("status") != "run_complete":
                raise RuntimeError("Joint trainer exited without completing the backbone stage")
            write_json(done, {"stage": name, "finished_at": time.time(), "artifact": artifact})
            print(f"Completed {name}", flush=True)
        if config.get("pipeline_mode") == "joint_backbone":
            write_json(progress_path, {"status": "backbone_complete", "finished_at": time.time(),
                                      "checkpoint": str(run_dir / "06_joint/best.pt"),
                                      "pending_stages": ["token_kda", "retrieval", "reranker", "fusion", "final_test"]})
            print("Backbone training complete; token KDA and RAG stages remain pending", flush=True)
        else:
            write_json(progress_path, {"status": "complete", "finished_at": time.time(), "metrics": str(run_dir / "07_evaluation" / "metrics.json")})
            print("Experiment complete", flush=True)


if __name__ == "__main__":
    main()
