MY_CMD="python main.py --config ./configs/default_cifar10_configs.py --eval_folder ./eval --mode test --workdir ./results"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='2' $MY_CMD
