def count_param(model, param_name, model_name='transformer'):

    model_num = 0
    for name, param in model.named_parameters():
        model_num += param.numel() if model_name in name else 0

    param_num = 0
    for name, param in model.named_parameters():
        param_num += param.numel() if param_name in name else 0

    return param_num / model_num
