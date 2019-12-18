# PyTorch Android Debug

Repo used for testing and debugging PyTorch for Android.

After cloning you need to put the file 'model.pt' into the assets folder. Run the following code in the base directory:

```
import torch
import torchvision

model = torchvision.models.inception_v3(pretrained=True)
model.eval()
example = torch.rand(1, 3, 299, 299)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/model.pt")
```
