from PIL import Image, ImageDraw
from otovision.preprocessing import crop_dark_border

def test_crop_dark_border_reduces_canvas():
    image=Image.new("RGB",(100,100),"black"); draw=ImageDraw.Draw(image); draw.rectangle((20,20,80,80),fill=(120,80,60)); cropped=crop_dark_border(image,threshold=12,padding=0); assert cropped.width<image.width; assert cropped.height<image.height
