from gymnasium.envs.registration import register

from environment.custom_env import MaternalReferralEnv

register(id="MaternalReferral-v0", entry_point="environment.custom_env:MaternalReferralEnv")

__all__ = ["MaternalReferralEnv"]
