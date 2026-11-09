# SPDX-License-Identifier: MIT AND Apache-2.0
# LatentSkill modifications: Copyright (c) 2026 Aofan Yu
# LatentSkill modifications are licensed under MIT (see LICENSE).
# Third-party portions remain licensed under Apache-2.0; see
# third_party/licenses/Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
# Based on and adapted from SHINE: [https://github.com/MuLabPKU/SHINE](https://github.com/MuLabPKU/SHINE)
# Modified by the LatentSkill authors in 2026.

def freeze_backbone_except_memory(backbone):
    """Freeze backbone parameters while keeping memory tokens trainable."""
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.model.mem_tokens.requires_grad = True
