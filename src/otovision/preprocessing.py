from __future__ import annotations

import numpy as np
from PIL import Image


def crop_dark_border(image: Image.Image, threshold: int = 12, padding: int = 4) -> Image.Image:
    img=image.convert("RGB"); arr=np.asarray(img); mask=arr.max(axis=2)>threshold; ys,xs=np.where(mask)
    if len(xs)==0 or len(ys)==0: return img
    left=max(int(xs.min())-padding,0); right=min(int(xs.max())+padding+1,img.width); top=max(int(ys.min())-padding,0); bottom=min(int(ys.max())+padding+1,img.height)
    if (right-left)<img.width*0.45 or (bottom-top)<img.height*0.45: return img
    return img.crop((left,top,right,bottom))
