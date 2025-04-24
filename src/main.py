import argparse
from src.pusher.env import PusherEnv
from src.ant_model.env import AntEnv
from src.humanoid_walking.env import HumanoidEnv

def main(args):
    if args.model == "pusher":
        env = PusherEnv(args.config, args.weights, args.mode)
    elif args.model == "ant":
        env = AntEnv(args.config, args.weights, args.mode)
    elif args.model == "humanoid":
        env = HumanoidEnv(args.config, args.weights, args.mode)
    else:
        raise ValueError("Invalid model type. Choose from 'pusher', 'ant', or 'humanoid'.") 
    
    if args.train:
        env.train(args.path)
        env.test(args.path)
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