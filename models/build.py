from .GET import GET


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    if model_type == 'GET':
        model = GET(patch_size=config.MODEL.GET.PATCH_SIZE,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dim=config.MODEL.GET.EMBED_DIM,
                    depths=config.MODEL.GET.DEPTHS,
                    num_heads=config.MODEL.GET.NUM_HEADS,
                    window_size=config.MODEL.GET.WINDOW_SIZE,
                    mlp_ratio=config.MODEL.GET.MLP_RATIO,
                    drop_rate=config.MODEL.DROP_RATE,
                    attn_drop_rate=config.MODEL.DROP_RATE // 2,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                    embed_split=config.MODEL.GET.EMBED_SPLIT,
                    group_num=config.MODEL.GET.GROUP_NUM,
                    )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
