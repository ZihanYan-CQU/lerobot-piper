"""RLT observations and actions are already normalized by the frozen PI05 processors."""

from lerobot.processor import make_default_pre_post_processors

from .configuration_rlt_actor import RLTActorConfig


def make_rlt_actor_pre_post_processors(config: RLTActorConfig, dataset_stats=None):
    if dataset_stats:
        raise ValueError("RLT features are already normalized; use the Stage 1 processors")
    return make_default_pre_post_processors(config, dataset_stats=None)
