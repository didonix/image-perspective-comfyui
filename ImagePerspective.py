import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 500
import numpy as np
import torch
from skimage import transform

class ImagePerspective:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "image" : ("IMAGE", {}),
                "left_side": ("FLOAT", {"default": 1, "min": 0.5, "max": 2, "step": 0.05}),
                "right_side": ("FLOAT", {"default": 1, "min": 0.5, "max": 2, "step": 0.05}),
                "top_side": ("FLOAT", {"default": 1, "min": 0.5, "max": 2, "step": 0.05}),
                "bottom_side": ("FLOAT", {"default": 1, "min": 0.5, "max": 2, "step": 0.05}),
                "fill_empty_space": (["constant", "symmetric", "wrap", "edge"], {"default": "constant"})
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "perspective"
    CATEGORY = "test"
   
    def perspective(self, image: torch.Tensor, left_side, right_side, top_side, bottom_side, fill_empty_space):
        batch_size, height, width, channels = image.shape
        width_t, height_t = (width / 100), (height / 100)
        bottom_w_diff = (width * bottom_side - width) / 2
        top_w_diff = (width * top_side - width) / 2
        left_h_diff = (height * left_side - height) / 2
        right_h_diff = (height * right_side - height) / 2
        tle1, tle2= (0 - top_w_diff), (0 - left_h_diff)
        tre1, tre2 = (width + top_w_diff), (0 - right_h_diff)
        bre1, bre2 = (width + bottom_w_diff), (height + right_h_diff)
        ble1, ble2 = (0 - bottom_w_diff), (height + left_h_diff)
        src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        end_points = np.array([[tle1, tle2], [tre1, tre2], [bre1, bre2], [ble1, ble2]])
        "fill_empty_space" == fill_empty_space
        img_out = torch.zeros_like(image)
        for b in range(batch_size):
            img = image[b].squeeze().numpy().astype(np.float32)
            tform = transform.estimate_transform("projective", src_points, end_points)
            img = transform.warp(img, tform.inverse, mode = str(fill_empty_space))
            plt.figure(num=None, figsize=(width_t, height_t), dpi=100)
            plt.cla()
            #plt.close(fig)
            transformed_img = (torch.from_numpy(img))
            img_out[b] = transformed_img.unsqueeze(0)
        return (img_out,)
    
NODE_CLASS_MAPPINGS = {
    "ImagePerspective": ImagePerspective,
}