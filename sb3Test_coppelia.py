from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from customEnviroment import Doggy_walker   

env = Doggy_walker()
check_env(env)  # Verifica se o ambiente segue as regras do Gym

model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)

# Treinar
model.learn(total_timesteps=10_000_000)