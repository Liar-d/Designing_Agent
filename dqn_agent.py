from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def train_dqn_agent(env):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./dqn_checkpoints/",
        name_prefix="dqn_model"
    )

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        exploration_final_eps=0.3,
        verbose=1
    )
    model.learn(total_timesteps=50000, callback=checkpoint_callback)
    model.save("dqn_final_model")
    return model


def run_dqn(model, env):
    obs, _ = env.reset()
    path = [tuple(obs["agent"])]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        path.append(tuple(obs["agent"]))
        env.render()
        if terminated:
            break
    return path
