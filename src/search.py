import argparse
from datetime import datetime
import yaml
import optuna
from optuna.storages import JournalStorage
from optuna.storages import JournalFileStorage
import os
import uuid
from src.base_env import BaseEnv

MODEL_ENV_MAP = {
    "pusher": "Pusher-v5",
    "ant": "Ant-v5",
    "humanoid": "Humanoid-v5",
    "cheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "humanoidstandup": "HumanoidStandup-v5",
    "invertedpendulum": "InvertedPendulum-v5",
    "inverteddoublependulum": "InvertedDoublePendulum-v5",
    "reacher": "Reacher-v5",
    "swimmer": "Swimmer-v5",
    "walker": "Walker2d-v5",
}

def build_cfg_for_trial(trial, trial_dir: str, base_cfg_path: str) -> str:
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["actor_lr"]  = trial.suggest_categorical(
        "actor_lr",
        [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2],
    )
    cfg["critic_lr"] = trial.suggest_categorical(
        "critic_lr",
        [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2],
    )

    cfg["td3_exploration_start"] = trial.suggest_float(
        "td3_exploration_start",
        0.1, 0.4,
        step=0.05,
    )
    cfg["td3_exploration_decay"] = trial.suggest_float(
        "td3_exploration_decay",
        0.95, 0.999,
        step=0.001,
    )

    cfg["target_update_freq"]  = trial.suggest_int("target_update_freq", 1, 10)
    cfg["actor_update_freq"]   = trial.suggest_int("actor_update_freq", 1, 10)

    out_cfg = os.path.join(
        trial_dir,
        f"pusher_cfg_t{trial.number}_{uuid.uuid4().hex[:6]}.yaml"
    )
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as fp:
        yaml.safe_dump(cfg, fp)

    return out_cfg

def objective(trial, mode: str, model: str, study_dir: str, base_cfg_path: str) -> float:
    cfg_path = build_cfg_for_trial(trial, study_dir, base_cfg_path)

    env_name = MODEL_ENV_MAP.get(model)
    if env_name is None:
        raise ValueError(f"Invalid model type '{model}'. Choose from: {', '.join(MODEL_ENV_MAP.keys())}")
    
    env = BaseEnv(config=cfg_path, weights=None, mode=mode, env_name=env_name)
    reward = env.train(os.path.join("search", f"{env_name}_{mode}", f"trial_{trial.number}"))

    return reward

def main(args):
    os.makedirs(args.study_dir, exist_ok=True)
    study_name = f"pusher_{args.mode}_{str(datetime.now()).replace(':', '-')}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    storage = JournalStorage(JournalFileStorage(os.path.join(args.study_dir, "journal.log")))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=2025),
        pruner=pruner,
    )
    
    study.optimize(
        lambda trial: objective(trial, args.mode, args.model, args.study_dir, args.base_cfg),
        n_trials     = args.n_trials,
        timeout      = args.timeout,
        n_jobs       = 1,
    )
    
    best = study.best_trial
    print(f"\n✅  Best reward  : {best.value:.2f}")
    print("✅  Params:")
    for k, v in best.params.items():
        print(f"   • {k:25s}: {v}")

    with open(os.path.join(args.study_dir, "best_params.yaml"), "w") as fp:
        yaml.safe_dump(best.params, fp)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=3600, help="timeout in seconds")
    parser.add_argument("--mode", choices=["TD3", "SAC"], default="TD3")
    parser.add_argument("--model", choices=list(MODEL_ENV_MAP.keys()), default="pusher")
    parser.add_argument("--study-dir", default="runs", help="directory to store study results")
    parser.add_argument("--base-cfg", default="cfg/pusher.yaml", help="base config file path")
    args = parser.parse_args()

    main(args)