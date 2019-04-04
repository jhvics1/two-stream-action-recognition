import argparse
import cv2
import glob

from time import sleep
from utils import *
from network import *
import dataloader


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

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
        #im = cv2.resize(im,(224,224))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,'Hello World!', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('Action Recognition',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sleep(0.1)

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()    