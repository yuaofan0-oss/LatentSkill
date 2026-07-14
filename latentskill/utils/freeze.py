def freeze_backbone_except_memory(backbone):
    """Freeze backbone parameters while keeping memory tokens trainable."""
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.model.mem_tokens.requires_grad = True
