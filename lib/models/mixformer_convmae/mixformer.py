from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_

from lib.utils.misc import is_main_process
from lib.models.mixformer_cvt.head import build_box_head
from lib.models.mixformer_cvt.utils import to_2tuple
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_vit.pos_utils import get_2d_sincos_pos_embed


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_test(self, x, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def forward_test(self, x, s_h, s_w):
        x = x + self.drop_path1(self.attn.forward_test(self.norm1(x), s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        x = x + self.drop_path1(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size_s=288, img_size_t=128, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.norm = norm_layer(embed_dim[-1])

        self.apply(self._init_weights)

        self.grid_size_s = img_size_s // (patch_size[0] * patch_size[1] * patch_size[2])
        self.grid_size_t = img_size_t // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim[2]), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim[2]), requires_grad=False)

        self.init_pos_embed()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        ### conv embeddings for x_t
        x_t = self.patch_embed1(x_t)
        x_t = self.pos_drop(x_t)
        for blk in self.blocks1:
            x_t = blk(x_t)
        x_t = self.patch_embed2(x_t)
        for blk in self.blocks2:
            x_t = blk(x_t)
        x_t = self.patch_embed3(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1) #BCHW --> BNC
        x_t = self.patch_embed4(x_t)

        ### conv embeddings for x_ot
        x_ot = self.patch_embed1(x_ot)
        x_ot = self.pos_drop(x_ot)
        for blk in self.blocks1:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed2(x_ot)
        for blk in self.blocks2:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed3(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        ### conv embeddings for x_s
        x_s = self.patch_embed1(x_s)
        x_s = self.pos_drop(x_s)
        for blk in self.blocks1:
            x_s = blk(x_s)
        x_s = self.patch_embed2(x_s)
        for blk in self.blocks2:
            x_s = blk(x_s)
        x_s = self.patch_embed3(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)

        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk(x, H_t, W_t, H_s, W_s)

        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d

    def forward_test(self, x_s):
        x_s = self.patch_embed1(x_s)
        x_s = self.pos_drop(x_s)
        for blk in self.blocks1:
            x_s = blk(x_s)
        x_s = self.patch_embed2(x_s)
        for blk in self.blocks2:
            x_s = blk(x_s)
        x_s = self.patch_embed3(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)

        H_s = W_s = self.grid_size_s
        x_s = x_s + self.pos_embed_s
        x_s = self.pos_drop(x_s)

        for blk in self.blocks3:
            x_s = blk.forward_test(x_s, H_s, W_s)

        x_s = rearrange(x_s, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return self.template, x_s

    def set_online(self, x_t, x_ot):
        ### conv embeddings for x_t
        x_t = self.patch_embed1(x_t)
        x_t = self.pos_drop(x_t)
        for blk in self.blocks1:
            x_t = blk(x_t)
        x_t = self.patch_embed2(x_t)
        for blk in self.blocks2:
            x_t = blk(x_t)
        x_t = self.patch_embed3(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
        x_t = self.patch_embed4(x_t)

        ### conv embeddings for x_ot
        x_ot = self.patch_embed1(x_ot)
        x_ot = self.pos_drop(x_ot)
        for blk in self.blocks1:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed2(x_ot)
        for blk in self.blocks2:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed3(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        B, C = x_t.size(0), x_t.size(-1)
        H_t = W_t = self.grid_size_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t


def get_mixformer_convmae(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == 'convmae_base':
        vit = ConvViT(
        img_size_s=img_size_s, img_size_t=img_size_t, patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif config.MODEL.VIT_TYPE == 'convmae_large':
        vit = ConvViT(
        img_size_s=img_size_s, img_size_t=img_size_t, patch_size=[4, 2, 2], embed_dim=[384, 768, 1024], depth=[2, 2, 20], num_heads=16, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'convmae_base' or 'convmae_large'")


    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_token' not in k:
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return vit


class MixFormer(nn.Module):
    def __init__(self, backbone, box_head, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type

    def forward(self, template, online_template, search, run_score_head=False, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search = self.backbone(template, online_template, search)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def forward_test(self, search, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone.forward_test(search)
        # search (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out_dict = {'pred_boxes': outputs_coord_new}
            return out_dict, outputs_coord_new
        else:
            raise KeyError


def build_mixformer_convmae(cfg, train=True) -> MixFormer:
    backbone = get_mixformer_convmae(cfg, train)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    model = MixFormer(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model
