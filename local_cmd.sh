MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10_eval_sample.py --eval_folder eval --mode eval_samples --workdir ./results --pytorch_dataset --test_sample_input_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/official_checkpoint/samples_1000x32x32x3.npz"

# /egr/research-dselab/renjie3/renjie/improved-diffusion/results/official_checkpoint/samples_1000x32x32x3.npz
# /egr/research-dselab/renjie3/renjie/improved-diffusion/results/130/model050000/samples_400x32x32x3.npz
# /egr/research-dselab/renjie3/renjie/improved-diffusion/results/cifar10_bird.npy

# MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10_bpd.py --eval_folder eval --mode eval --workdir ./results --pytorch_dataset"

# MY_CMD="python main.py --config ./configs/vp/ddpm/cifar10.py --eval_folder eval --mode train --workdir ./results --pytorch_dataset"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='3' $MY_CMD
