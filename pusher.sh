python3 -m src.main --config src/pusher/config.yaml --mode TD3 --path outputs/pusher/TD3/ --model pusher --train
xvfb-run --server-args="-screen 0 1024x768x24" python3 -m src.main --config src/pusher/config.yaml --mode TD3 --path outputs/ --model pusher --weights outputs/pusher/TD3/
python3 -m src.main --config src/pusher/config.yaml --mode SAC --path outputs/ --model pusher --weights outputs/pusher/SAC/ --test
xvfb-run --server-args="-screen 0 1024x768x24" python3 -m src.main --config src/pusher/config.yaml --mode SAC --path outputs/ --model pusher --weights outputs/pusher/SAC/