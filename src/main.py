import argparse
from src.base_env import BaseEnv

def main(args):
    if args.model == "pusher":
        env_name = "Pusher-v5"
    elif args.model == "ant":
        env_name = "Ant-v5"
    elif args.model == "humanoid":
        env_name = "Humanoid-v5"
    else:
        raise ValueError("Invalid model type. Choose from 'pusher', 'ant', or 'humanoid'.") 
    
    env = BaseEnv(args.config, args.weights, args.mode, env_name)
    if args.train:
        env.train(args.path)
    else: 
        env.test(args.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training for various MuJoCo models")
    parser.add_argument("--model", type=str, choices=["pusher", "ant", "humanoid"], required=True, help="Model type to train/test")
    parser.add_argument("--config", type=str, default="default", help="Configuration name")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--mode", type=str, default="train", choices=["TD3", "SAC"], help="Mode of operation")
    parser.add_argument("--path", type=str, default="./results", help="Path to save/load models and results")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
                        
    args = parser.parse_args()
    main(args)