set -x

# ======== 模型路径 ========
# 使用 SFT 后的 cold-start 3B 模型作为 GRPO 的起点
# MODEL_PATH=../SFT/output/v3-20260614-170158/checkpoint-1197
MODEL_PATH=../SFT/models/Qwen2.5-VL-3B-Instruct

# ======== 输出路径 ========
EXP_NAME=qwen2_5_vl_3b_grpo
PROJECT_NAME=rl-with-cold-start
SAVE_PATH=./checkpoints/${PROJECT_NAME}/${EXP_NAME}

# ======== SwanLab 输出目录 ========
export SWANLAB_DIR=./swanlab_log

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/Multimodal-RL-Data/data \
    data.val_files=./data/geometry3k/data/test-00000-of-00001.parquet \
    data.rollout_batch_size=512 \
    data.val_batch_size=500 \
    data.max_pixels=1204224 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.n=10 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.reward.score_function=./examples/score_function/math.py:compute_score \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=2 \
    trainer.save_freq=20 \
    trainer.save_limit=3 \
    trainer.val_freq=10 \
    trainer.val_before_train=true \
    trainer.save_checkpoint_path=${SAVE_PATH} \

