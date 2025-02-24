from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env, ObservationWrapper, RewardWrapper, spaces
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

import mujoco
from PIL import Image
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE  # noqa

from .wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
    RenderMission,
)


TASK_NAME_TO_ENV_ID = {
    "atari-alien": "ALE/Alien-v5",
    "atari-amidar": "ALE/Amidar-v5",
    "atari-assault": "ALE/Assault-v5",
    "atari-asterix": "ALE/Asterix-v5",
    "atari-asteroids": "ALE/Asteroids-v5",
    "atari-atlantis": "ALE/Atlantis-v5",
    "atari-bankheist": "ALE/BankHeist-v5",
    "atari-battlezone": "ALE/BattleZone-v5",
    "atari-beamrider": "ALE/BeamRider-v5",
    "atari-berzerk": "ALE/Berzerk-v5",
    "atari-bowling": "ALE/Bowling-v5",
    "atari-boxing": "ALE/Boxing-v5",
    "atari-breakout": "ALE/Breakout-v5",
    "atari-centipede": "ALE/Centipede-v5",
    "atari-choppercommand": "ALE/ChopperCommand-v5",
    "atari-crazyclimber": "ALE/CrazyClimber-v5",
    "atari-defender": "ALE/Defender-v5",
    "atari-demonattack": "ALE/DemonAttack-v5",
    "atari-doubledunk": "ALE/DoubleDunk-v5",
    "atari-enduro": "ALE/Enduro-v5",
    "atari-fishingderby": "ALE/FishingDerby-v5",
    "atari-freeway": "ALE/Freeway-v5",
    "atari-frostbite": "ALE/Frostbite-v5",
    "atari-gopher": "ALE/Gopher-v5",
    "atari-gravitar": "ALE/Gravitar-v5",
    "atari-hero": "ALE/Hero-v5",
    "atari-icehockey": "ALE/IceHockey-v5",
    "atari-jamesbond": "ALE/Jamesbond-v5",
    "atari-kangaroo": "ALE/Kangaroo-v5",
    "atari-krull": "ALE/Krull-v5",
    "atari-kungfumaster": "ALE/KungFuMaster-v5",
    "atari-montezumarevenge": "ALE/MontezumaRevenge-v5",
    "atari-mspacman": "ALE/MsPacman-v5",
    "atari-namethisgame": "ALE/NameThisGame-v5",
    "atari-phoenix": "ALE/Phoenix-v5",
    "atari-pitfall": "ALE/Pitfall-v5",
    "atari-pong": "ALE/Pong-v5",
    "atari-privateeye": "ALE/PrivateEye-v5",
    "atari-qbert": "ALE/Qbert-v5",
    "atari-riverraid": "ALE/Riverraid-v5",
    "atari-roadrunner": "ALE/RoadRunner-v5",
    "atari-robotank": "ALE/Robotank-v5",
    "atari-seaquest": "ALE/Seaquest-v5",
    "atari-skiing": "ALE/Skiing-v5",
    "atari-solaris": "ALE/Solaris-v5",
    "atari-spaceinvaders": "ALE/SpaceInvaders-v5",
    "atari-stargunner": "ALE/StarGunner-v5",
    "atari-surround": "ALE/Surround-v5",
    "atari-tennis": "ALE/Tennis-v5",
    "atari-timepilot": "ALE/TimePilot-v5",
    "atari-tutankham": "ALE/Tutankham-v5",
    "atari-upndown": "ALE/UpNDown-v5",
    "atari-venture": "ALE/Venture-v5",
    "atari-videopinball": "ALE/VideoPinball-v5",
    "atari-wizardofwor": "ALE/WizardOfWor-v5",
    "atari-yarsrevenge": "ALE/YarsRevenge-v5",
    "atari-zaxxon": "ALE/Zaxxon-v5",
    "babyai-action-obj-door": "BabyAI-ActionObjDoor-v0",
    "babyai-blocked-unlock-pickup": "BabyAI-BlockedUnlockPickup-v0",
    "babyai-boss-level-no-unlock": "BabyAI-BossLevelNoUnlock-v0",
    "babyai-boss-level": "BabyAI-BossLevel-v0",
    "babyai-find-obj-s5": "BabyAI-FindObjS5-v0",
    "babyai-go-to-door": "BabyAI-GoToDoor-v0",
    "babyai-go-to-imp-unlock": "BabyAI-GoToImpUnlock-v0",
    "babyai-go-to-local": "BabyAI-GoToLocal-v0",
    "babyai-go-to-obj-door": "BabyAI-GoToObjDoor-v0",
    "babyai-go-to-obj": "BabyAI-GoToObj-v0",
    "babyai-go-to-red-ball-grey": "BabyAI-GoToRedBallGrey-v0",
    "babyai-go-to-red-ball-no-dists": "BabyAI-GoToRedBallNoDists-v0",
    "babyai-go-to-red-ball": "BabyAI-GoToRedBall-v0",
    "babyai-go-to-red-blue-ball": "BabyAI-GoToRedBlueBall-v0",
    "babyai-go-to-seq": "BabyAI-GoToSeq-v0",
    "babyai-go-to": "BabyAI-GoTo-v0",
    "babyai-key-corridor": "BabyAI-KeyCorridor-v0",
    "babyai-mini-boss-level": "BabyAI-MiniBossLevel-v0",
    "babyai-move-two-across-s8n9": "BabyAI-MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "BabyAI-OneRoomS8-v0",
    "babyai-open-door": "BabyAI-OpenDoor-v0",
    "babyai-open-doors-order-n4": "BabyAI-OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "BabyAI-OpenRedDoor-v0",
    "babyai-open-two-doors": "BabyAI-OpenTwoDoors-v0",
    "babyai-open": "BabyAI-Open-v0",
    "babyai-pickup-above": "BabyAI-PickupAbove-v0",
    "babyai-pickup-dist": "BabyAI-PickupDist-v0",
    "babyai-pickup-loc": "BabyAI-PickupLoc-v0",
    "babyai-pickup": "BabyAI-Pickup-v0",
    "babyai-put-next-local": "BabyAI-PutNextLocal-v0",
    "babyai-put-next": "BabyAI-PutNextS7N4-v0",
    "babyai-synth-loc": "BabyAI-SynthLoc-v0",
    "babyai-synth-seq": "BabyAI-SynthSeq-v0",
    "babyai-synth": "BabyAI-Synth-v0",
    "babyai-unblock-pickup": "BabyAI-UnblockPickup-v0",
    "babyai-unlock-local": "BabyAI-UnlockLocal-v0",
    "babyai-unlock-pickup": "BabyAI-UnlockPickup-v0",
    "babyai-unlock-to-unlock": "BabyAI-UnlockToUnlock-v0",
    "babyai-unlock": "BabyAI-Unlock-v0",
    "metaworld-assembly": "assembly-v2",
    "metaworld-basketball": "basketball-v2",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-button-press-topdown-wall": "button-press-topdown-wall-v2",
    "metaworld-button-press-topdown": "button-press-topdown-v2",
    "metaworld-button-press-wall": "button-press-wall-v2",
    "metaworld-button-press": "button-press-v2",
    "metaworld-coffee-button": "coffee-button-v2",
    "metaworld-coffee-pull": "coffee-pull-v2",
    "metaworld-coffee-push": "coffee-push-v2",
    "metaworld-dial-turn": "dial-turn-v2",
    "metaworld-disassemble": "disassemble-v2",
    "metaworld-door-close": "door-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-open": "door-open-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-drawer-close": "drawer-close-v2",
    "metaworld-drawer-open": "drawer-open-v2",
    "metaworld-faucet-close": "faucet-close-v2",
    "metaworld-faucet-open": "faucet-open-v2",
    "metaworld-hammer": "hammer-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "metaworld-handle-press-side": "handle-press-side-v2",
    "metaworld-handle-press": "handle-press-v2",
    "metaworld-handle-pull-side": "handle-pull-side-v2",
    "metaworld-handle-pull": "handle-pull-v2",
    "metaworld-lever-pull": "lever-pull-v2",
    "metaworld-peg-insert-side": "peg-insert-side-v2",
    "metaworld-peg-unplug-side": "peg-unplug-side-v2",
    "metaworld-pick-out-of-hole": "pick-out-of-hole-v2",
    "metaworld-pick-place-wall": "pick-place-wall-v2",
    "metaworld-pick-place": "pick-place-v2",
    "metaworld-plate-slide-back-side": "plate-slide-back-side-v2",
    "metaworld-plate-slide-back": "plate-slide-back-v2",
    "metaworld-plate-slide-side": "plate-slide-side-v2",
    "metaworld-plate-slide": "plate-slide-v2",
    "metaworld-push-back": "push-back-v2",
    "metaworld-push-wall": "push-wall-v2",
    "metaworld-push": "push-v2",
    "metaworld-reach-wall": "reach-wall-v2",
    "metaworld-reach": "reach-v2",
    "metaworld-shelf-place": "shelf-place-v2",
    "metaworld-soccer": "soccer-v2",
    "metaworld-stick-pull": "stick-pull-v2",
    "metaworld-stick-push": "stick-push-v2",
    "metaworld-sweep-into": "sweep-into-v2",
    "metaworld-sweep": "sweep-v2",
    "metaworld-window-close": "window-close-v2",
    "metaworld-window-open": "window-open-v2",
    "mujoco-ant": "Ant-v4",
    "mujoco-doublependulum": "InvertedDoublePendulum-v4",
    "mujoco-halfcheetah": "HalfCheetah-v4",
    "mujoco-hopper": "Hopper-v4",
    "mujoco-humanoid": "Humanoid-v4",
    "mujoco-pendulum": "InvertedPendulum-v4",
    "mujoco-pusher": "Pusher-v4",
    "mujoco-reacher": "Reacher-v4",
    "mujoco-standup": "HumanoidStandup-v4",
    "mujoco-swimmer": "Swimmer-v4",
    "mujoco-walker": "Walker2d-v4",
}


def get_task_names() -> List[str]:
    """
    Get all the environment ids.

    Returns:
        list: List of environment ids
    """
    return list(TASK_NAME_TO_ENV_ID.keys())


class AtariDictObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {"image_observation": spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)}
        )

    def observation(self, observation):
        observation = np.transpose(observation, (1, 2, 0))  # make channel last
        return {"image_observation": observation}


def make_atari(task_name: str, episodic_life: bool = True, clip_reward: bool = True, **kwargs) -> Env:
    kwargs = {"frameskip": 1, "repeat_action_probability": 0.0, **kwargs}
    if task_name == "atari-montezumarevenge":
        kwargs["max_episode_steps"] = 18_000
    env = gym.make(TASK_NAME_TO_ENV_ID[task_name], **kwargs)
    env.metadata["render_fps"] = 30
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = NumpyObsWrapper(env)
    env = AtariDictObservationWrapper(env)
    return env


class BabyAIDictObservationWrapper(ObservationWrapper):
    """
    Wrapper for BabyAI environments.

    Flatten the pseudo-image and concatenante it to the direction observation.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        n_image = self.observation_space["image"].high.flatten()
        n_direction = self.observation_space["direction"].n
        self.observation_space = spaces.Dict(
            {
                "text_observation": env.observation_space.spaces["mission"],
                "discrete_observation": spaces.MultiDiscrete([n_direction, *n_image]),
            }
        )

    def observation(self, observation: Dict[str, np.ndarray]):
        discrete_observation = np.append(observation["direction"], observation["image"].flatten())
        return {
            "text_observation": observation["mission"],
            "discrete_observation": discrete_observation,
        }


class FloatRewardWrapper(RewardWrapper):
    def reward(self, reward):
        return float(reward)


def make_babyai(task_name: str, **kwargs) -> Env:
    env = gym.make(TASK_NAME_TO_ENV_ID[task_name], **kwargs)
    env = BabyAIDictObservationWrapper(env)
    env = FloatRewardWrapper(env)
    env = RenderMission(env)
    return env


class ContinuousObservationDictWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"continuous_observation": env.observation_space})

    def observation(self, observation):
        return {"continuous_observation": observation}

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, 
                 env_name: str,
                 img_height: int = 128,
                 img_width: int = 128,
                 cameras=('corner2',),
                 env_kwargs=None,):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](**env_kwargs)
        env._freeze_rand_vec = False
        super().__init__(env)
        # # I don't know why
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        shape_meta = {
            'action_dim': 4,
            'observation': {
                'rgb': {'corner_rgb': (3, 128, 128)},
                'lowdim': {'robo_states': 8},
            },
            'task': {
                'type': 'onehot',
                'n_tasks': 45
            }
        }

        self.img_width = img_width
        self.img_height = img_height
        obs_meta = shape_meta['observation']
        self.rgb_outputs = list(obs_meta['rgb'])
        self.lowdim_outputs = list(obs_meta['lowdim'])

        self.cameras = cameras
        self.viewer = OffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta['lowdim'][key],),
                dtype=np.float32
            )
        obs_space_dict['obs_gt'] = env.observation_space
        self.observation_space = gym.spaces.Dict(obs_space_dict)

    def resize_image(image: np.ndarray, size: Tuple[int, int] = (84, 84)) -> np.ndarray:
        # Convert numpy array to PIL Image, resize, and convert back
        pil_img = Image.fromarray(image)
        resized = np.array(pil_img.resize(size, Image.Resampling.BICUBIC))
        return resized

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = super().step(action)
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt

        next_obs = self.make_obs(obs_gt)

        terminated = info['success'] == 1
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs_gt, info = super().reset()
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt

        obs = self.make_obs(obs_gt)

        return obs, info

    def make_obs(self, obs_gt):
        obs = {}
        # obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
        # obs['obs_gt'] = obs_gt

        # image_dict = {}
        # for camera_name in self.cameras:
        #     image_obs = self.render(camera_name=camera_name, mode='all')
        #     image_dict[camera_name] = image_obs
        # for key in self.rgb_outputs:
        #     obs[key] = image_dict[f'{key[:-4]}2'][::-1] # since generated dataset at the time had corner key instead of corner2
        obs['image_observation'] = self.resize_image(self.render(camera_name=self.cameras[0], mode='all').copy()[::-1])

        return obs

    def render(self, camera_name=None, mode='rgb_array'):
        if camera_name is None:
            camera_name = self.cameras[0]
        cam_id = mujoco.mj_name2id(self.env.model, 
                                mujoco.mjtObj.mjOBJ_CAMERA, 
                                camera_name)
        
        return self.viewer.render(
            render_mode=mode,
            camera_id=cam_id
        )
    
    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.viewer.close()


def make_metaworld(task_name: str, **kwargs) -> Env:
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{TASK_NAME_TO_ENV_ID[task_name]}-goal-observable"]
    env = ContinuousObservationDictWrapper(env_cls(**kwargs))
    return env

def make_pixel_metaworld(task_name: str, **kwargs) -> Env:
    return MetaWorldWrapper(env_name=TASK_NAME_TO_ENV_ID[task_name])


def make_mujoco(task_name: str, **kwargs) -> Env:
    env = gym.make(TASK_NAME_TO_ENV_ID[task_name], **kwargs)
    env = ContinuousObservationDictWrapper(env)
    return env


def make(task_name: str, **kwargs) -> Env:
    """
    Make an environment from the task name.

    Args:
        task_name (`str`):
            The name of the task to make. Check `get_task_names()` for the list of available tasks.

    Raises:
        ValueError:
            If the task name is not in the list of available tasks.

    Returns:
        Env: The environment.
    """
    if task_name.startswith("atari"):
        return make_atari(task_name, **kwargs)

    elif task_name.startswith("babyai"):
        return make_babyai(task_name, **kwargs)

    elif task_name.startswith("metaworld"):
        return make_metaworld(task_name, **kwargs)

    elif task_name.startswith("mujoco"):
        return make_mujoco(task_name, **kwargs)
    else:
        raise ValueError(f"Unknown task name: {task_name}. Available task names: {get_task_names()}")
