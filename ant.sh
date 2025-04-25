python3 -m src.main --config src/ant_model/config.yaml --mode TD3 --path outputs/ant_model/TD3 --model ant --train
xvfb-run --server-args="-screen 0 1024x768x24" python3 -m src.main --config src/ant_model/config.yaml --mode TD3 --path outputs/ --model ant --weights outputs/ant_model/TD3/
python3 -m src.main --config src/ant_model/config.yaml --mode SAC --path outputs/ant_model/SAC --model ant --train
xvfb-run --server-args="-screen 0 1024x768x24" python3 -m src.main --config src/ant_model/config.yaml --mode SAC --path outputs/ --model ant --weights outputs/ant_model/SAC/