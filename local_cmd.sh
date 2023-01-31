MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10_bpd.py --eval_folder eval --mode eval --workdir ./results"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='5' $MY_CMD
