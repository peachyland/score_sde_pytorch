# MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10_bpd.py --eval_folder eval --mode eval --workdir ./results"


MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10.py --eval_folder eval --mode train --workdir ./results --hard_examples"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0' $MY_CMD
