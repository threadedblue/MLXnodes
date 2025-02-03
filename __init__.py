from .txt2img import MLXText2Image
from .img2img import MLXImg2Img

NODE_CLASS_MAPPINGS = {
    "MLXText2Image": MLXText2Image,
    "MLXImg2Img" : MLXImg2Img,
}
