dataset=data/ultrapose
check_dir=checkpoints
finetune_path=checkpoints/transUltra.pth
bsPerGPU=2
threads=4
gpus=0,1,2,3
gpu_num=4
dataset_name="coco"
model_name="transformer"
lr=1e-4
visname="TransUltra"
cpe=10
group_name=${visname}
disturl="tcp://localhost:54322"
ws=${gpu_num}

CUDA_VISIBLE_DEVICES=${gpus} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env train.py \
  --load_pair --check_per_epoch ${cpe} --visdom_name ${visname} --dist_url ${disturl} --dataset_name ${dataset_name} \
  --model_name ${model_name} --dataroot ${dataset} --use_visdom --batch_size ${bsPerGPU} --checkpoints_dir ${check_dir} \
  --num_threads ${threads} --lr ${lr} --world_size=${ws} --finetune_model_path ${finetune_path} \
  --distributed_run --group_name ${group_name}
