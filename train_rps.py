import ray
from ray import tune
from rps_env import RPSExternalEnv

ray.init()

tune.run(
    "PPO",
    stop={"training_iteration": 1},
    config={
        "enable_connectors": False,
        "env": RPSExternalEnv,
        "framework": "torch",
        "num_workers": 1,
        "env_config": {},
    },
)
