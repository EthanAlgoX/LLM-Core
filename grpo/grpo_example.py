import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# ==============================================================================
# GRPO (Group Relative Policy Optimization) 示例脚本
# ==============================================================================
# GRPO 是 DeepSeek-V3/R1 中使用的强化学习算法，无需原型的 Critic 网络，
# 通过组内样本的相对得分来计算优势，非常适合算力受限的环境。

def grpo_training_demo():
    # 1. 基础配置 (使用 Qwen3-0.6B 作为演示)
    model_id = "Qwen/Qwen3-0.6B"
    output_dir = "qwen3_grpo_out"
    print(f"Using model: {model_id}")
    
    # 2. 准备奖励函数 (Reward Functions)
    print("Defining reward functions...")
    def length_reward_func(completions, **kwargs):
        """奖励长度适中的回答"""
        return [0.5 if 20 <= len(c) <= 100 else 0.0 for c in completions]

    def format_reward_func(completions, **kwargs):
        """奖励包含特定格式的回答 (模拟 R1 的 <think> 标签校验)"""
        rewards = []
        for content in completions:
            if "<think>" in content and "</think>" in content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # 3. 加载数据集 (这里使用一个简单的演示数据集)
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": ["dummy_data.jsonl"]}, split="train") 
    
    # 4. 训练配置 (针对 Mac MPS 优化)
    print("Setting up GRPO config...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=1,
        bf16=True,                   # Mac MPS 推荐
        num_generations=4,           # GRPO 的组大小 (Group Size)
        max_prompt_length=256,
        max_completion_length=256,
        report_to="tensorboard",
    )

    # 5. 初始化 Trainer
    print("Initializing GRPOTrainer (this may take a while)...")
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[length_reward_func, format_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    # 6. 开始训练
    print("Starting GRPO training loop...")
    trainer.train()

if __name__ == "__main__":
    # 准备一个极简的演示数据文件
    with open("dummy_data.jsonl", "w") as f:
        f.write('{"prompt": "请用一段话描述什么是人工智能，并使用 <think> 标签包裹您的思考过程。"}\n')
        f.write('{"prompt": "解释一下为什么天空是蓝色的。"}\n')
    
    grpo_training_demo()
