import argparse
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

def main(args):
    env_name = MODEL_ENV_MAP.get(args.model)
    if env_name is None:
        raise ValueError(f"Invalid model type '{args.model}'. Choose from: {', '.join(MODEL_ENV_MAP.keys())}")
    
    env = BaseEnv(args.config, args.weights, args.mode, env_name)
    if args.train:
        env.train(args.path)
    else: 
        env.test(args.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training for various MuJoCo models")
    parser.add_argument("--model", type=str, choices=list(MODEL_ENV_MAP.keys()), required=True, help="Model type to train/test")
    parser.add_argument("--config", type=str, default="default", help="Configuration name")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--mode", type=str, default="train", choices=["TD3", "SAC"], help="Mode of operation")
    parser.add_argument("--path", type=str, default="./results", help="Path to save/load models and results")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
                        
    args = parser.parse_args()
    main(args)
