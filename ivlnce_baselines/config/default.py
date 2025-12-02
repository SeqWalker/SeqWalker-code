from typing import List, Optional, Union
import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN
from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()
_C.CMD_TRAILING_OPTS = []
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = ["disk"]
_C.VIDEO_DIR = "data/videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"

_C.EVAL = CN()
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.USE_CKPT_CONFIG = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.ITERATIVE_MAP_RESET = "iterative"
_C.EVAL.ITERATIVE_GT_PATHS = "data/gt_ndtw.json"

_C.IL = CN()
_C.IL.lr = 1e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4

_C.IL.use_iw = True
_C.IL.inflection_weight_coef = 3.2
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
_C.IL.is_requeue = False

_C.IL.DAGGER = CN()

_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.lmdb_map_size = 1.0e13
_C.IL.DAGGER.lmdb_fp16 = False
_C.IL.DAGGER.lmdb_commit_frequency = 500
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
_C.IL.DAGGER.drop_existing_lmdb_features = True

_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER = CN()
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.resolution_meters = 0.1
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.height_clip = 0.1
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.height_meters = 6.4
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.width_meters = 6.4

_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"

_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_map = False
_C.MODEL.ablate_instruction = False
_C.MODEL.tour_memory = False
_C.MODEL.tour_memory_variant = False
_C.MODEL.memory_at_end = False
_C.MODEL.train_unrolled = False
_C.MODEL.disable_tour_memory = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = True

_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256
_C.MODEL.RGB_ENCODER.trainable = False

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)
_C.MODEL.DEPTH_ENCODER.trainable = False

_C.MODEL.SEMANTIC_MAP_ENCODER = CN()
_C.MODEL.SEMANTIC_MAP_ENCODER.classname = "SemanticMapEncoder"
_C.MODEL.SEMANTIC_MAP_ENCODER.num_semantic_classes = 13
_C.MODEL.SEMANTIC_MAP_ENCODER.output_size = 256
_C.MODEL.SEMANTIC_MAP_ENCODER.channels = 32
_C.MODEL.SEMANTIC_MAP_ENCODER.last_ch_mult = 4
_C.MODEL.SEMANTIC_MAP_ENCODER.trainable = True
_C.MODEL.SEMANTIC_MAP_ENCODER.from_pretrained = False
_C.MODEL.SEMANTIC_MAP_ENCODER.checkpoint = ""
_C.MODEL.SEMANTIC_MAP_ENCODER.custom_lr = False
_C.MODEL.SEMANTIC_MAP_ENCODER.lr = 2.5e-6

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
