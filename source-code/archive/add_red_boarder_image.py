from PIL import Image, ImageOps
# pip install pillow
# Pass in an image and get an image with a red boarder
ImageOps.expand(Image.open('data/yellowcup_180.jpeg'), border=20, fill="#f00").save('data/yellowcup_180_holding.jpeg')