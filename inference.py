import torch
import importlib
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as transforms
from lang_encoder import FrozenTextEncoder, DEFAULT_LANG_MODEL


class letterbox():
    def __init__(self, size=256, fill=128):
        self.size = size
        self.fill = fill  # padding color, 0 for black

    def __call__(self, img: Image.Image):
        w, h = img.size

        # 计算缩放比例
        scale = self.size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 计算padding
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill)
        return img
    
def preprocess(img1, img2, obs, tac1, tac2, yaml_config, letterbox_flag=False):
    # norm obs
    obs = torch.tensor(obs, dtype=torch.float32)
    norm_type = yaml_config['data']['norm_type']
    if norm_type == 'mean_std':
        obs_mean = torch.tensor(yaml_config['data']['observation_mean'])
        obs_std = torch.tensor(yaml_config['data']['observation_std']).clamp_min(1e-8)
        obs = (obs - obs_mean) / obs_std
    else:
        obs_min = torch.tensor(yaml_config['data']['observation_min'])
        obs_max = torch.tensor(yaml_config['data']['observation_max'])
        obs = (obs - obs_min) / (obs_max - obs_min)
        obs = obs.clamp(0, 1.0)
    obs = obs.unsqueeze(0) # (b, 28)

    # norm tactile
    tac_max_l = yaml_config['data']['tac_left_max']
    tac_max_r = yaml_config['data']['tac_right_max']
    tac1 = torch.from_numpy(tac1 / tac_max_l).float().clamp(0, 1.0)
    tac2 = torch.from_numpy(tac2 / tac_max_r).float().clamp(0, 1.0)
    tac1, tac2 = tac1.unsqueeze(0), tac2.unsqueeze(0)

    # preprocess image
    img_config = yaml_config['img']
    if letterbox_flag:
        resize = letterbox(img_config['img_size'][0])
    else:
        resize = transforms.Resize(img_config['img_size'])
    test_transform = transforms.Compose([
        resize,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=img_config['img_mean'],
            std=img_config['img_std'])
    ]) 
    img1 = Image.fromarray(img1)
    img1 = test_transform(img1).unsqueeze(0) # pil2tensor and unsqueeze (b, 3, h, w)
    img2 = Image.fromarray(img2)
    img2 = test_transform(img2).unsqueeze(0) # pil2tensor and

    return img1, img2, obs, tac1, tac2


def postprocess(action, yaml_config):
    norm_type = yaml_config['data']['norm_type']
    if norm_type == 'mean_std':
        action_mean = torch.tensor(yaml_config['data']['action_mean'])
        action_std = torch.tensor(yaml_config['data']['action_std']).clamp_min(1e-8)
        action = action * action_std[None, :] + action_mean[None, :]
    else:
        action_min = torch.tensor(yaml_config['data']['action_min'])
        action_max = torch.tensor(yaml_config['data']['action_max'])
        action = action * (action_max - action_min)[None, :] + action_min[None, :]
    return action


def _prep_image_tensor(img: np.ndarray, img_size, img_mean, img_std) -> torch.Tensor:
    """HWC uint8 (or float) numpy image -> normalized CHW float tensor, no augmentation.
    Shared by both RGB cameras and the 4 vision-based tactile streams -- same normalization
    used in lerobot_dataset.py's rgb_transform/tactile_transform (minus the training-only
    color jitter / dropout).
    """
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(list(img_size)),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    return transform(img)


_VITAC_TACTILE_KEYS_ORDER = ("tactile_left_0", "tactile_right_0", "tactile_left_1", "tactile_right_1")

_lang_encoder_cache = {}


def _get_lang_encoder(model_name=DEFAULT_LANG_MODEL, device="cpu") -> FrozenTextEncoder:
    # Loading the tokenizer/model is slow; keep one frozen instance around across calls.
    key = (model_name, device)
    if key not in _lang_encoder_cache:
        _lang_encoder_cache[key] = FrozenTextEncoder(model_name=model_name, device=device)
    return _lang_encoder_cache[key]


def preprocess_vitac(img1, img2, obs, yaml_config, tactile_imgs: dict = None):
    """Preprocessing for the deco_vitac model family (models/deco_vitac/).

    Args:
        img1, img2: HWC numpy arrays (observation.images.camera0/1)
        obs: 1D array-like, length model.obs_dim (20 for the ManiSkill-ViTac 2026 contract)
        tactile_imgs: optional dict with keys tactile_left_0/tactile_right_0/tactile_left_1/
            tactile_right_1 -> HWC numpy arrays. Required iff yaml_config['model']['use_tactile'].
    """
    data_cfg = yaml_config['data']
    obs = torch.tensor(obs, dtype=torch.float32)
    obs_mean = torch.tensor(data_cfg['observation_mean'])
    obs_std = torch.tensor(data_cfg['observation_std']).clamp_min(1e-8)
    obs = ((obs - obs_mean) / obs_std).unsqueeze(0)  # (1, obs_dim)

    img_cfg = yaml_config['img']
    img1_t = _prep_image_tensor(img1, img_cfg['img_size'], img_cfg['img_mean'], img_cfg['img_std']).unsqueeze(0)
    img2_t = _prep_image_tensor(img2, img_cfg['img_size'], img_cfg['img_mean'], img_cfg['img_std']).unsqueeze(0)

    use_tactile = yaml_config['model'].get('use_tactile', False)
    if use_tactile:
        if tactile_imgs is None:
            raise ValueError("model.use_tactile is True but no tactile_imgs were provided")
        frames = [
            _prep_image_tensor(tactile_imgs[k], img_cfg['img_size'], img_cfg['img_mean'], img_cfg['img_std'])
            for k in _VITAC_TACTILE_KEYS_ORDER
        ]
        tactile_t = torch.stack(frames, dim=0).unsqueeze(0)  # (1, n_sensors, 3, H, W)
    else:
        h, w = img_cfg['img_size']
        tactile_t = torch.zeros(1, 4, 3, h, w)

    return img1_t, img2_t, obs, tactile_t


def predict_action_vitac(model, device, yaml_config, img1, img2, obs, prompt: str, tactile_imgs: dict = None, lang_model_name=DEFAULT_LANG_MODEL):
    """Run one flow-matching inference pass for a models/deco_vitac model.

    `prompt` is embedded on the fly with the same frozen text encoder used at training
    time (rather than looked up in the training-time cache), since the deployed prompt may
    not have been seen during training.
    """
    with torch.no_grad():
        img1_t, img2_t, obs_t, tactile_t = preprocess_vitac(img1, img2, obs, yaml_config, tactile_imgs=tactile_imgs)
        img1_t, img2_t, obs_t, tactile_t = img1_t.to(device), img2_t.to(device), obs_t.to(device), tactile_t.to(device)

        encoder = _get_lang_encoder(model_name=lang_model_name, device="cpu")
        lang_embed = encoder.embed(prompt).to(device)  # (1, lang_embed_dim)

        action = model(img1_t, img2_t, obs=obs_t, act=None, lang_embed=lang_embed, tactile_imgs=tactile_t, action_mask=None, training=False)
        action = action.cpu().squeeze(0)  # (chunksize, act_dim)
        action = postprocess(action, yaml_config)

    return action


def predict_action(model, device, yaml_config, img1, img2, obs, task_idx=0, tac1=None, tac2=None):
    task_idx = torch.tensor(task_idx, dtype=torch.long).unsqueeze(0).to(device)
    if yaml_config['img']['img_size'] == [256, 256]:
        letterbox = True
    else:
        letterbox = False
    with torch.no_grad():
        img1, img2, obs, tac1, tac2 = preprocess(img1, img2, obs, tac1, tac2, yaml_config, letterbox_flag=letterbox)
        img1, img2, obs, tac1, tac2 = img1.to(device), img2.to(device), obs.to(device), tac1.to(device), tac2.to(device)
        action = model(img1, img2, obs=obs, act=None, task_idx=task_idx, tac1=tac1, tac2=tac2, action_mask=None, training=False)
        action = action.cpu().squeeze(0) # (1, chunksize, dim) --> (chunksize, dim)
        action = postprocess(action, yaml_config)  # (chunksize, dim)
        
    return action


def modeling(yaml_config): 
    model_name = yaml_config['model_name']
    importmodule = importlib.import_module(f"models.{model_name}")
    model = importmodule.modeling(**yaml_config['model']) 
    return model

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:

        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

