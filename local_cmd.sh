MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10_bpd.py --eval_folder eval --mode eval --workdir ./results --pytorch_dataset"


# MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10.py --eval_folder eval --mode train --workdir ./results --pytorch_dataset"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='5' $MY_CMD
