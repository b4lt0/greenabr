from gymnasium.envs.registration import register

register(
     id="greenabr/greenabr-v0",
     entry_point="greenabr.envs:GreenABREnv",
)