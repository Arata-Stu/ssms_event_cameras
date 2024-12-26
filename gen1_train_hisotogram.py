import os

# 固定パラメータ
gpu_ids = 0
batch_size_per_gpu = 5
train_workers_per_gpu = 6
eval_workers_per_gpu = 2
mdl_cfg = "tiny"  # MDL_CFGの値を指定
data_dir = "/media/arata22/AT_2TB/pre_gen1"  # DATA_DIRの値を指定

event_frame_dts = [5, 10, 20, 50, 100]  # 必要に応じて値を追加

# ループ処理
for dt in event_frame_dts:
    command = f"""
    python3 RVT/train.py model=rnndet dataset=gen1 dataset.path={data_dir} wandb.project_name=RVT_{dt} \
    wandb.group_name=gen1 +experiment/gen1={mdl_cfg}.yaml hardware.gpus={gpu_ids} \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'stacked_histogram_dt={dt}_nbins=10'"
    """
    print(f"Running command for event_frame_dt={dt}_nbins=10")
    os.system(command)  # 実際にコマンドを実行
