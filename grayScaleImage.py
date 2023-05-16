from imports import *

#================================================================
#
# Function: saveJson(currentJson, newMetaData)
#
# Description: This function is just a wrapper for the open cv
#              grayscaling procedure since it has to be able to 
#              catch any opencv exceptions, and exception handling
#              in main makes it cluttered and hard to read.
#
# Returns: image | openCV image
#
#================================================================
def grayScaleImage(image):

    #FIXME: ADD EXCEPTION HANDLING
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image