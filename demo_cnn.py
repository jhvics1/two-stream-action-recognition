import argparse
import cv2
import glob

from time import sleep
from utils import *
from network import *
import dataloader



parser = argparse.ArgumentParser(description='DEMO application')
parser.add_argument('--path', default='', type=str, metavar='PATH', help='path to the video file to play (default: none)')

def main():

    # 1. Get Video (Class) type & its path to play
    # 2. Get path of the Optical flow images of given video file
    # 3. Create a thread where opencv will run video play
    # 4. Create a thread where motion_cnn will run prediction
    # 5. Send prediction result to the thread which plays video
    # 6. Put/show prediction result on the video play window.
    #
    global arg
    arg = parser.parse_args()
    print arg

    images = [file for file in glob.glob('{0}/*jpg'.format(arg.path))]
    images.sort()
    print ('size of images : {0}'.format(len(images)))

    for image in images:
        im = cv2.imread(image)
        blur = cv2.GaussianBlur(im,(0,0),1)
        cv2.imshow('Action Recognition',blur)
        cv2.imwrite('MyPic.jpg', blur)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sleep(0.2)


if __name__=='__main__':
    main()    