from .resnet_model import create_resnet_model
from .cnn_model import create_cnn_model
from .svm_model import create_svm_model
from .transformer_model import create_transformer_model
from .resent_mixstyle import create_resnet_mixstyle_model

def create_model(model_type, num_classes=4, **kwargs):
    """
    创建指定类型的模型
    """
    if model_type == 'resnet':
        return create_resnet_model(num_classes, **kwargs)
    elif model_type == 'cnn':
        return create_cnn_model(num_classes, **kwargs)
    elif model_type == 'svm':
        return create_svm_model(**kwargs)
    elif model_type == 'transformer':
        return create_transformer_model(num_classes, **kwargs)
    elif model_type == 'resnetmixstyle':
        return create_resnet_mixstyle_model(num_classes, mix="crossdomain", **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")