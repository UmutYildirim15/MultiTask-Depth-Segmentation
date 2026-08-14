# VidEoMT-ADAS -> a query-based ViT adapted for temporal multi-task prediction (segmentation + depth) on consecutive SYNTHIA frames.
# Unlike the other baselines, this one carries a set of learned queries from frame t to frame t+1 (see `prev_queries`), giving it a lightweight form of
# temporal context without a dedicated video encoder. It needs a paired-frame dataset to train.


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

DEFAULT_VIT = "vit_base_patch16_224"
DEFAULT_IMG_SIZE = 512
DEFAULT_EMBED_DIM = 768


class VidEoMT_ADAS(nn.Module):

    def __init__(self, num_classes: int = 19, embed_dim: int = DEFAULT_EMBED_DIM,
                 img_size: int = DEFAULT_IMG_SIZE):
        super().__init__()
        self.vit = timm.create_model(DEFAULT_VIT, pretrained=True, num_classes=0, img_size=img_size)
        self.patch_size = 16
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.init_queries = nn.Parameter(torch.randn(1, num_classes + 1, embed_dim))
        self.mask_proj = nn.Linear(embed_dim, embed_dim)
        self.depth_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, prev_queries=None):
        b = x.shape[0]
        x = self.vit.patch_embed(x)
        x = x + self.vit.pos_embed[:, 1:, :]

        queries = self.init_queries.expand(b, -1, -1) if prev_queries is None else prev_queries
        cls_token = self.vit.cls_token.expand(b, -1, -1)

        x = torch.cat((cls_token, queries, x), dim=1)
        x = self.vit.pos_drop(x)

        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)

        num_queries = queries.shape[1]
        query_out = x[:, 1:1 + num_queries, :]
        patch_out = x[:, 1 + num_queries:, :]

        semantic_queries = query_out[:, :self.num_classes, :]
        depth_query = query_out[:, self.num_classes:, :]

        patch_semantic = self.mask_proj(patch_out)
        patch_depth = self.depth_proj(patch_out)

        seg_logits = torch.bmm(semantic_queries, patch_semantic.transpose(1, 2))
        depth_logits = torch.bmm(depth_query, patch_depth.transpose(1, 2))

        feat_size = self.img_size // self.patch_size
        seg_logits = seg_logits.view(b, self.num_classes, feat_size, feat_size)
        depth_logits = depth_logits.view(b, 1, feat_size, feat_size)

        seg_logits = F.interpolate(seg_logits, size=(self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False)
        depth_logits = F.interpolate(depth_logits, size=(self.img_size, self.img_size),
                                      mode="bilinear", align_corners=False)

        # query_out is returned so the caller can pass it as 'prev_queries'
        # for the next frame in the sequence
        return seg_logits, depth_logits, query_out