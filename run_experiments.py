import argparse
import numpy as np
import matplotlib.pyplot as plt
from environment import CityGridEnv
from dqn_agent import train_dqn_agent, run_dqn
import time


def visualize_comparison(env, dqn_path):
    plt.figure(figsize=(16, 16))

    # DQN path
    plt.subplot(1, 2, 1)
    img = np.ones((env.size, env.size, 3))
    for obs in env.obstacles:
        img[obs] = [0, 0, 0]
    for step in dqn_path:
        img[step] = [1, 0, 0] if step != env.goal else [0, 1, 0]
    plt.imshow(img)
    plt.title(f"DQN Path (Length: {len(dqn_path)})")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Run mode: train (train DQN) or test (test only)")
    args = parser.parse_args()

    env = CityGridEnv(size=5, obstacle_ratio=0.1)

    start_time = time.time()

    if args.mode == "train":
        print("=== Training DQN ===")
        model = train_dqn_agent(env)
    else:
        from stable_baselines3 import DQN
        model = DQN.load("dqn_final_model")

    elapsed_time = time.time() - start_time
    print(f"Training time consumption: {elapsed_time:.2f} second")

    print("=== Running DQN ===")
    dqn_path = run_dqn(model, env)

    print(f"\nDQN Path Length: {len(dqn_path)}")
    visualize_comparison(env, dqn_path)


if __name__ == "__main__":
    main()