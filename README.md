# Pytorch-Distributed-Temple
A temple for distributed training of pytorch.

Training command:
CUDA_VISIBLE_DEVICES=gpu_id python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=your_port train.py -c "./config/config.json"

Testing command:
CUDA_VISIBLE_DEVICES=gpu_id python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=your_port test.py -c "./config/config.json" -r your_resume_path
