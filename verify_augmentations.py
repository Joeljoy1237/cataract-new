
import torch
import os
from torchvision import transforms
from PIL import Image
from preprocessing.augmentations import get_train_transforms

def save_image(tensor, path):
    # Tensor is (C, H, W). Convert to (H, W, C) and [0, 255]
    img = tensor.permute(1, 2, 0).numpy()
    # If image was normalized in pipeline (which we simulated), it is [0, 1].
    # But for visualization we assume input is [0, 1].
    img = (img * 255).astype('uint8')
    Image.fromarray(img).save(path)

def verify():
    os.makedirs('test_outputs', exist_ok=True)
    
    # Create a dummy image (stripes to see rotation/flips)
    img = torch.zeros(3, 224, 224)
    img[:, :112, :] = 1.0 # Top half white
    img[:, :, :112] = 0.5 # Left half gray blend
    
    print("Saving original dummy image...")
    save_image(img, 'test_outputs/original.png')
    
    # Fundus Augmentations
    print("Testing Fundus Transforms...")
    fundus_transforms = get_train_transforms('fundus')
    for i in range(3):
        aug_img = fundus_transforms(img)
        save_image(aug_img, f'test_outputs/fundus_aug_{i}.png')
        
    # Slit Lamp Augmentations
    print("Testing Slit Lamp Transforms...")
    slit_transforms = get_train_transforms('slit_lamp')
    for i in range(3):
        aug_img = slit_transforms(img)
        save_image(aug_img, f'test_outputs/slit_aug_{i}.png')
        
    print("Verification complete. Check test_outputs/ folder.")

if __name__ == "__main__":
    verify()
