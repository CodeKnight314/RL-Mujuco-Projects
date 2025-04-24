import argparse
from datetime import datetime
import yaml
import optuna
from optuna.storages import JournalStorage
from optuna.storages import JournalFileStorage
import os
import uuid

from src.pusher.env import PusherEnv


def build_cfg_for_trial(trial, trial_dir: str, base_cfg_path: str) -> str:
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    cfg["actor_lr"]  = trial.suggest_float("actor_lr", 5e-5, 1e-2, log=True)
    cfg["critic_lr"] = trial.suggest_float("critic_lr", 5e-5, 1e-2, log=True)

    cfg["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.5, 2.5)

    cfg["td3_exploration_start"] = trial.suggest_float("td3_exploration_start", 0.1, 0.4)
    cfg["td3_exploration_decay"] = trial.suggest_float("td3_exploration_decay", 0.95, 0.999)
    
    cfg["target_update_freq"] = trial.suggest_int("target_update_freq", 1, 20)
    cfg["actor_update_freq"] = trial.suggest_int("actor_update_freq", 1, 20)
    cfg["reward_scale"] = trial.suggest_int("reward_scale", 1, 10)
    cfg["reward_clip"] = trial.suggest_float("reward_clip", 5.0, 20.0)

    out_cfg = os.path.join(trial_dir, f"pusher_cfg_t{trial.number}_{uuid.uuid4().hex[:6]}.yaml")
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as fp:
        yaml.safe_dump(cfg, fp)
    
    return out_cfg

def objective(trial, mode: str, study_dir: str, base_cfg_path: str) -> float:
    cfg_path = build_cfg_for_trial(trial, study_dir, base_cfg_path)
    
    env = PusherEnv(cfg_path, weights=None, mode=mode)
    
    reward = env.train("outputs/")
    
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
        lambda trial: objective(trial, args.mode, args.study_dir, args.base_cfg),
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
    parser.add_argument("--n-trials", type=int, default=50,
                        help="number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="timeout in seconds")
    parser.add_argument("--mode", choices=["TD3", "SAC"], default="TD3")
    parser.add_argument("--study-dir", default="runs",
                        help="directory to store study results")
    parser.add_argument("--base-cfg", default="cfg/pusher.yaml",
                        help="base config file path")
    args = parser.parse_args()

    main(args)