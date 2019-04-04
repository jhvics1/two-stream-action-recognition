import argparse
import cv2
import glob
import random

from time import sleep
from utils import *
from network import *
import dataloader

from motion_cnn import *

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20,200)
fontScale              = 1
fontColor              = (255*random.random(), 255*random.random(), 255*random.random())
lineType               = 2

parser = argparse.ArgumentParser(description='DEMO application')
parser.add_argument('--path', default='', type=str, metavar='PATH', help='path to the video file to play (default: none)')
parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='model_motion/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

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

    model = get_prediction(arg)

    #images = [file for file in glob.glob('{0}/*jpg'.format(arg.path))]
    #images.sort()
    #print ('size of images : {0}'.format(len(images)))
    #for image in images:
    #    im = cv2.imread(image)
    #    #im = cv2.resize(im,(224,224))
    #    cv2.putText(im,'Hello World!', 
    #                bottomLeftCornerOfText, 
    #                font, 
    #                fontScale,
    #                fontColor,
    #                lineType)
    #    cv2.imshow('Action Recognition',im)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #    sleep(0.1)
    #cv2.destroyAllWindows()

def get_prediction(arg):
    data_loader = dataloader.Motion_DataLoader(
                        BATCH_SIZE=32,
                        num_workers=8,
                        path='../dataset/tvl1_flow/',
                        ucf_list='UCF_list/',
                        ucf_split='04',
                        in_channel=10,
                        )
    
    train_loader,test_loader, test_video = data_loader.run()
    #Model 
    model = Motion_CNN(
                        # Data Loader
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=0,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 10*2,
                        test_video=test_video
                        )
    #Training
    model.get_prediction(arg.path)


if __name__=='__main__':
    main()    