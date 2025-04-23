import argparse
from src.pusher.env import PusherEnv

def main(args):
    if args.model == "pusher":
        env = PusherEnv(args.config, args.weights, args.mode)
        env.train(args.path)
        env.test(args.path)
    elif args.model == "ant":
        pass 
    elif args.model == "humanoid":
        pass
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training for various MuJoCo models")
    parser.add_argument("--model", type=str, choices=["pusher", "ant", "humanoid"], required=True, help="Model type to train/test")
    parser.add_argument("--config", type=str, default="default", help="Configuration name")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--mode", type=str, default="train", choices=["TD3", "SAC"], help="Mode of operation")
    parser.add_argument("--path", type=str, default="./results", help="Path to save/load models and results")
                        
    args = parser.parse_args()
    main(args)