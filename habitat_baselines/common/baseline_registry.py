

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):

        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):

        from habitat import RLEnv

        return cls._register_impl("env", to_register, name, assert_type=RLEnv)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)

    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):

        from habitat_baselines.rl.ppo.policy import Policy

        return cls._register_impl(
            "policy", to_register, name, assert_type=Policy
        )

    @classmethod
    def get_policy(cls, name: str):

        return cls._get_impl("policy", name)

    @classmethod
    def register_obs_transformer(
        cls, to_register=None, *, name: Optional[str] = None
    ):

        from habitat_baselines.common.obs_transformers import (
            ObservationTransformer,
        )

        return cls._register_impl(
            "obs_transformer",
            to_register,
            name,
            assert_type=ObservationTransformer,
        )

    @classmethod
    def get_obs_transformer(cls, name: str):

        return cls._get_impl("obs_transformer", name)


baseline_registry = BaselineRegistry()
