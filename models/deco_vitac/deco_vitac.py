import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.deco.img_encoder import ResNet34
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import apply_rotary_emb, RotaryPosEmbed
from models.deco.deco import MMAttention, timeEmb
from models.deco_vitac.tactile_img_encoder import TactileImageEncoder
import einops


class DECOVitac(nn.Module):
    """DECO adapted for the ManiSkill-ViTac 2026 / official LeRobot data contract.

    Differences from models.deco.deco.DECO:
      - `obs_dim` is independent from `act_dim` (original DECO ties the proprioception
        encoder's input width to action_dim, which breaks once state/action are no longer
        the same size).
      - Tactile input is 4 vision-based RGB streams (TactileImageEncoder) instead of DECO's
        original 1062-dim per-hand region scalars.
      - Conditioning uses a cached/frozen text-embedding of the task prompt
        (`lang_embed`) instead of a one-hot task index, since ManiSkill-ViTac tasks are
        specified by free-form language rather than a fixed small task set.
    MMAttention (joint self-attention + tactile cross-attention + LoRA plugin adapter) is
    reused unchanged from models.deco.deco -- it already treats tactile as an arbitrary
    [B, N, dim] token set, so nothing about the two-stage vision->plugin design needs to change.
    """

    def __init__(
        self,
        act_dim,
        obs_dim,
        chunk_size,
        obs_state=True,
        use_tactile=False,
        plugin=False,
        plugin_rank=32,
        use_language_condition=True,
        lang_embed_dim=384,
        tactile_pool_size=2,
        tactile_t_hist=1,
        inf_step=10,
        img_pretrain=False,
        num_attn_blocks=6,
        heads=8,
        dim=512,
        rope_axes_dim=[256, 256],
        freeze_backbone=True,
    ):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_language_condition = use_language_condition
        self.inference_step = inf_step
        self.rope = RotaryPosEmbed(head_dim, rope_axes_dim)
        self.img_encoder = ResNet34()  # shared img encoder, ImageNet-pretrained
        self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)

        self.pos_idx_embedd = nn.Embedding(2, dim)  # distinguish camera0 vs camera1
        if self.obs_state:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )

        if self.use_tactile:
            self.tactile_encoder = TactileImageEncoder(dim=dim, pool_size=tactile_pool_size, t_hist=tactile_t_hist)

        if self.use_language_condition:
            self.lang_proj = nn.Sequential(
                nn.Linear(lang_embed_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )

        self.time_embedd = nn.Sequential(
            timeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.mmattn = nn.ModuleList(
            [MMAttention(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)]
        )
        self.linear = nn.Linear(dim, act_dim)  # final action prediction head

        if not plugin:
            self.initialize_weights()

        if img_pretrain:
            model_dict = torch.load(img_pretrain, map_location="cpu")
            pretrain_dict = {
                k: v
                for k, v in model_dict.items()
                if k in self.img_encoder.state_dict().keys() and np.shape(model_dict[k]) == np.shape(v)
            }
            self.img_encoder.load_state_dict(pretrain_dict, strict=True)
            if freeze_backbone:
                self.freeze()

    def forward(
        self,
        img1,
        img2,
        obs=None,
        act=None,
        lang_embed=None,
        tactile_imgs=None,
        action_mask=None,
        training=True,
    ):
        """
        Args:
            img1: [B, 3, H, W]  (observation.images.camera0)
            img2: [B, 3, H, W]  (observation.images.camera1)
            obs: [B, obs_dim]
            act: [B, chunk, act_dim]
            lang_embed: [B, lang_embed_dim]  frozen text-encoder embedding of the task prompt
            tactile_imgs: [B, n_sensors*t_hist, 3, H, W] or None
            training: bool
        """
        feat, image_rotary_emb = self.img_encoding(img1, img2)

        if self.use_tactile:
            tactile = self.tactile_encoder(tactile_imgs)
        else:
            tactile = None

        if self.obs_state:
            obs = self.obs_encoder(obs)
        if self.use_language_condition:
            lang_emb = self.lang_proj(lang_embed)

        if training:
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))
            act, noise = self.add_noise(act, t)
            t = self.time_embedd(t)
            if self.obs_state:
                t = t + obs
            if self.use_language_condition:
                t = t + lang_emb
            feat, act = self.atten_forward(feat, act, image_rotary_emb=image_rotary_emb, t=t, tactile=tactile)
            return act, noise

        else:
            sample = torch.randn(img1.shape[0], self.chunk_size, self.act_dim).to(img1.device)
            t = get_schedule(self.inference_step, self.chunk_size)
            for t_curr, t_prev in zip(t[:-1], t[1:]):
                t_vec = torch.full((img1.shape[0],), t_curr, dtype=img1.dtype, device=img1.device)
                t_vec = self.time_embedd(t_vec)
                if self.obs_state:
                    t_vec = t_vec + obs
                if self.use_language_condition:
                    t_vec = t_vec + lang_emb
                _, denoise_act = self.atten_forward(feat, sample, image_rotary_emb=image_rotary_emb, t=t_vec, tactile=tactile)
                sample = sample + (t_prev - t_curr) * denoise_act

            return sample

    def img_encoding(self, img1, img2):
        assert img1.shape == img2.shape, "img1 and img2 must have the same shape"
        img = torch.cat([img1, img2], dim=0)
        feat = self.img_encoder(img)
        feat = self.img_head(feat)
        feat1, feat2 = feat.chunk(2, dim=0)

        feat_h, feat_w = feat1.shape[-2:]
        image_rotary_emb = self.rope(feat_h, feat_w)

        feat1 = einops.rearrange(feat1, "b c h w -> b (h w) c")
        feat2 = einops.rearrange(feat2, "b c h w -> b (h w) c")
        img_id = torch.tensor([0] * feat1.shape[1] + [1] * feat2.shape[1]).to(img2.device)
        img_id = self.pos_idx_embedd(img_id).repeat(img1.shape[0], 1, 1)
        feat = torch.cat([feat1, feat2], dim=1)
        feat = feat + img_id

        return feat, image_rotary_emb

    def atten_forward(self, img, act, image_rotary_emb, t, tactile=None):
        act = self.action_encoder(act)
        act = act + self.action_embedd

        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb, tactile)

        act = self.linear(act)
        return img, act

    def add_noise(self, act: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(act).to(act.device)
        t = t.view(act.shape[0], 1, 1)
        act = (1 - t) * act + t * noise
        return act, noise

    def freeze(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)


def modeling(
    action_dim,
    obs_dim,
    chunk_size,
    obs_state,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_language_condition=True,
    lang_embed_dim=384,
    tactile_pool_size=2,
    tactile_t_hist=1,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    img_pretrain=None,
    freeze_backbone=True,
    pretrain_model_path=False,
    adapter_model_path=False,
):
    model = DECOVitac(
        act_dim=action_dim,
        obs_dim=obs_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        use_tactile=use_tactile,
        plugin=plugin,
        plugin_rank=plugin_rank,
        use_language_condition=use_language_condition,
        lang_embed_dim=lang_embed_dim,
        tactile_pool_size=tactile_pool_size,
        tactile_t_hist=tactile_t_hist,
        inf_step=inf_step,
        num_attn_blocks=num_attn_blocks,
        heads=heads,
        dim=dim,
        rope_axes_dim=rope_axes_dim,
        img_pretrain=img_pretrain,
        freeze_backbone=freeze_backbone,
    )

    if pretrain_model_path:
        if use_tactile:
            if plugin:
                if adapter_model_path:
                    print("loading adapter weights from {} for adapter inference".format(adapter_model_path))
                    model_dict = torch.load(adapter_model_path, map_location="cpu")
                    model.load_state_dict(model_dict, strict=True)
                else:
                    # stage 2: freeze the stage-1 vision-only weights, only train the
                    # tactile encoder + cross-attention + PI_Adapter (LoRA) params.
                    print("loading vision pretrained weights from {} for adapter finetuning".format(pretrain_model_path))
                    model_dict = torch.load(pretrain_model_path, map_location="cpu")
                    model_state = model.state_dict()
                    pretrain_dict = {
                        k: v for k, v in model_dict.items() if k in model_state.keys() and v.shape == model_state[k].shape
                    }
                    model.load_state_dict(pretrain_dict, strict=False)
                    for name, param in model.named_parameters():
                        if name in model_dict.keys():
                            param.requires_grad = False
                        else:
                            print("trainable params:", name)
            else:
                print("loading vision-tactile pretrained weights from {} for inference".format(pretrain_model_path))
                model_dict = torch.load(pretrain_model_path)
                model.load_state_dict(model_dict, strict=True)
        else:
            print("loading vision pretrained weights from {} for inference".format(pretrain_model_path))
            model_dict = torch.load(pretrain_model_path)
            model.load_state_dict(model_dict, strict=True)

    return model
