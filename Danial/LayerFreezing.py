def freeze_transformers(model):
    for param in model.transformer.parameters():
        param.requires_grad = False
    print("Transformer layers frozen.")
def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone layers frozen.")
def freeze_class_embed(model):
    for param in model.class_embed.parameters():
        param.requires_grad = False
    print("Class embedding layers frozen.")