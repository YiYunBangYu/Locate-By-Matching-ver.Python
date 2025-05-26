from cv2 import KAZE_create , cvtColor , COLOR_BGR2GRAY
import numpy as np

def summon(img):
    img = cvtColor(img , COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    kaze = KAZE_create()
    p , d = kaze.detectAndCompute(img , None)

    return p,d


