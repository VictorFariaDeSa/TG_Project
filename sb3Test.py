import gymnasium as gym
from stable_baselines3 import PPO
from helper import plot
from collections import deque

# Ambiente de treino (sem render)
env_train = gym.make("HalfCheetah-v5", render_mode=None)

# Cria o modelo PPO
model = PPO("MlpPolicy", env_train, verbose=1, learning_rate=1e-4)

# Treina o modelo
model.learn(total_timesteps=1_000_000)

# Fecha ambiente de treino
env_train.close()

print("EVALUATION -------------------------------------------------------------")
# Ambiente de avaliação (com renderização)
env_eval = gym.make("HalfCheetah-v5", render_mode="human")

# Avaliação com renderização e gráfico
obs, _ = env_eval.reset()
plot_scores = []
plot_mean_scores = []
last_scores = deque(maxlen=100)
score = 0

while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    score += reward
    env_eval.render()  # <- renderiza apenas aqui!
    
    if terminated or truncated:
        obs, _ = env_eval.reset()
        plot_scores.append(score)
        last_scores.append(score)
        mean_score = sum(last_scores) / len(last_scores)
        plot_mean_scores.append(mean_score)
        score = 0
        plot(plot_scores, plot_mean_scores)

env_eval.close()
