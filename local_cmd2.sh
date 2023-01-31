MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10.py --eval_folder cifar10 --mode train --workdir ./results"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='3,4' $MY_CMD
