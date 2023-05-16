#==============================================================================
#
# Class : CS 6420
#
# Author : Tyler Martin
#
# Project Name : Project 4 | K means clustering
# Date: 4/6/2023
#
# Description: This project implements a k means clustering method
#
# Notes: Since I know you prefer to read and work in C++, this file is set
#        up to mimic a standard C/C++ flow style, including a __main__()
#        declaration for ease of viewing. Also, while semi-colons are not 
#        required in python, they can be placed at the end of lines anyway, 
#        usually to denote a specific thing. In my case, they denote globals, 
#        and global access, just to once again make it easier to parse my code
#        and see what it is doing and accessing.
#
#==============================================================================

#"header" file imports
from imports import *
from checkImages import *
from grayScaleImage import *
from operators import *

#================================================================
#
# NOTES: THE OUTPATH WILL HAVE THE LAST / REMOVED IF IT EXISTS
#        THE imageType WILL HAVE A . APPLIED TO THE FRONT AFTER
#        CHECKING VALIDITY
#
#================================================================

rect_selector = None
fequency_representation_shifted = None
fig = None
filename = None
image = None
magnitude_spectrum = None
freq_filt_img = None
reset = False
sized = .50
label = "NIL"
center = "NIL"

#================================================================
#
# Function: findCenter(event)
#
# Description: This function uses the find countour function to
#              find the centroid of an object set
#
#================================================================
def findCenter(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if (cX != 0 and cY != 0):
            return np.array([cX, cY])

#================================================================
#
# Function: key(event)
#
# Description: This function simply listens for key events and
#              either saves the image, resets the canvas, or 
#              quits the program based on which valid key is passed
#
#================================================================
def key(event):

    global magnitude_spectrum
    global filename
    global freq_filt_img
    global reset

    if event.key == 'r' and reset:
        os.execv(sys.executable, ['python'] + sys.argv)
    elif event.key == 'q':
        sys.exit(0)

#================================================================
#
# Function: onclick(eclick, erelease)
#
# Description: This function listens for the drawing of the bounding
#              rect and then makes the mask and applies it to the 
#              image before performing idft and displaying the 
#              result.
#
#================================================================
def onclick(eclick, erelease):

    #globals
    global rect_selector
    global fequency_representation_shifted
    global fig
    global image
    global freq_filt_img
    global reset
    global sized

    #make mask
    img = np.copy(image)
    z = 0
    row, column, z = img.shape

    #mask
    reset = True

    #apply K-means clustering to find states
    flattened = np.float32(img.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret,label,center=cv2.kmeans(flattened, 25, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    labels = label.reshape((img.shape[:2]))

    #FIXME: combine these commands
    labels[labels > labels[int(rect_selector.extents[2])][int(rect_selector.extents[0])]] = 0
    labels[labels < labels[int(rect_selector.extents[2])][int(rect_selector.extents[0])]] = 0
    labels[labels == labels[int(rect_selector.extents[2])][int(rect_selector.extents[0])]] = 1

    #rebuild the image
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #try opening the image
    kernel = np.ones((20, 20), np.uint8)
    opening = cv2.dilate(res2, kernel, iterations=1)
    opening = cv2.threshold(cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    opening = opening[0:img.shape[0],0:img.shape[1]]

    #FIXME: the built-in flood fill method needs to be rewritten
    stateMask = np.zeros((row+2, column+2), np.uint8)
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    cv2.floodFill(opening, stateMask, (int(rect_selector.extents[0]), int(rect_selector.extents[2])), res2[int(rect_selector.extents[2])][int(rect_selector.extents[0])].tolist(), 1)
    stateMask = stateMask[0:row, 0:column]
    originalCenter = findCenter(stateMask)
    stateMask = cv2.threshold(cv2.cvtColor(cv2.bitwise_and(res2,res2,mask=stateMask), cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    extractedState = cv2.bitwise_and(img,img,mask=stateMask)
    
    resized_state = cv2.warpAffine(extractedState, np.float32([[2, 0, originalCenter[0] - 2*originalCenter[0]],[0, 2, originalCenter[1] - 2*originalCenter[1]]]), (extractedState.shape[1], extractedState.shape[0]))
    
    #remove background from resized state
    tmp = cv2.cvtColor(resized_state, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(resized_state)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    #show the new image
    plt.title('Altered Image\n[press \'r\' to reset the board, or \'q to quit\']'), plt.xticks([]), plt.yticks([])
    plt.imshow(dst, cmap = 'gray', alpha=0.6)


#================================================================
#
# Function: __main__
#
# Description: This function is the python equivalent to a main
#              function in C/C++ (added just for ease of your
#              reading, it has no practical purpose)
#
#================================================================

def __main__(argv):

    #globals
    global rect_selector
    global fequency_representation_shifted
    global fig
    global filename
    global image
    global magnitude_spectrum

    #variables that contain the command line switch
    #information
    inPath = "nothing"
    depth = 1
    mode = 1
    intensity = 1
    primary = "nothing"
    filename = "outImage"
    direction = 0

    # get arguments and parse
    try:
      opts, args = getopt.getopt(argv,"h:t:s:")

    except getopt.GetoptError:
            print("map_segment [-h] -t input_image ")
            print("===========================================================================================================")
            print("-t : Target Image (t)")
            #print("-s : Output Image (s)")
            sys.exit(2)
    for opt, arg in opts:

        if opt == ("-h"):
            print("map_segment [-h] -t input_image ")
            print("===========================================================================================================")
            print("-t : Target Image (t)")
            #print("-s : Output Image (s)")
            

        elif opt == ("-m"):
            if (int(arg) < 10 and int(arg) > 0):
                mode = int(arg)
            else:
                print("Invalid Mode Supplied. Only Values 1 through 9 Are Accepted.")

        elif opt == ("-t"):
            primary = arg
        elif opt == ("-s"):
            filename = arg

    #demand images if we are not supplied any
    if (primary == "nothing"):
        print("map_segment [-h] -t input_image ")
        print("===========================================================================================================")
        print("-t : Target Image (t)")
        #print("-s : Output Image (s)")
        sys.exit(2)

    

    #open the image
    image = checkImages(primary)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #keep windows reseeting upon request
    while (1):

        fig, ax = plt.subplots(1, 1, figsize=(10,7))
        plt.imshow(image, cmap = 'gray')
        plt.title('map'), plt.xticks([]), plt.yticks([])
        fig.canvas.mpl_connect('key_press_event', key)

        #set on draw listener
        rect_selector = RectangleSelector(ax, onclick, button=[1], minspanx=0, minspany=0)

        #display
        plt.show()

#start main
argv = ""
__main__(sys.argv[1:])