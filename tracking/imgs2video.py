import cv2
import os
from PIL import Image


def PicToVideo(imgPath, videoPath):

    images = os.listdir(imgPath)

    images.sort(key=lambda x: int(x[:-4]))

    fps = 20  # 帧率

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    im = Image.open(imgPath + images[0])

    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, im.size,isColor=True)

    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])
        videoWriter.write(frame)

    videoWriter.release()   # 释放

imgPath = '/data0/cyt/experiments/mixformer/results_vis/diver_results/'
videoPath = '/data0/cyt/experiments/mixformer/results_vis/diver_results.avi'

# imgPath = '/data0/cyt/jc/project/sequences/diver/color/'
# videoPath = '/data0/cyt/experiments/mixformer/results_vis/diver.avi'
PicToVideo(imgPath, videoPath)