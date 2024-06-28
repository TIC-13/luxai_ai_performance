import torch
import torchsummary

from torchvision.models.densenet import densenet121
from torchvision.models.mobilenetv2 import mobilenet_v2

def get_mobilenetv2():
    return mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V2")
    
def get_densenet121():
    return densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")

def get_model(name="mobilenet", output_size=1, fine_tune_classifier=False):
    model = None

    if   name == "mobilenet": 
        model = get_mobilenetv2()
        last_layer = model.classifier[-1]

    elif name == "densenet": 
        model = get_densenet121()
        last_layer = model.classifier
    
    else: 
        pass

    if model:
        prev_feats       = last_layer.weight.shape[1]
        model.classifier = torch.nn.Linear(prev_feats, output_size)

        if fine_tune_classifier:
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model



if __name__ == "__main__":
    
    device  = "cuda"

    model1  = get_model(name="mobilenet", output_size=2, fine_tune_classifier=True)
    model2  = get_model(name="densenet",  output_size=3, fine_tune_classifier=True)

    #model1.to(device)
    #model2.to(device)

    torchsummary.summary(model1, (3,224,224))
    torchsummary.summary(model2, (3,256,256))