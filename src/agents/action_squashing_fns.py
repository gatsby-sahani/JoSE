from jax import numpy as jnp


class ActionSquashing:
    def __init__(self):
        pass

    def add_function(self, name, func):
        setattr(self, name, func)


def tanh_squashing(env, cfg):

    action_squasher = ActionSquashing()

    action_squasher.pre_squash_clip = 1e6

    action_squasher.add_function("squashing_fn", lambda x: jnp.tanh(x))
    action_squasher.add_function(
        "squashing_fn_gradient",
        lambda x: 1.0 - jnp.tanh(x) ** 2,
    )

    return action_squasher


def get_action_squasher(env, cfg):

    action_squasher = tanh_squashing(env, cfg)

    return action_squasher
