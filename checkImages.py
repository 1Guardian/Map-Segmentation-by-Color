from imports import *

#================================================================
#
# Function: checkImages(path)
#
# Description: This function takes in the image Path supplied to
#              it and returns the image read in, or a boolean
#              with the value 'False' if the read fails
#
# Returns: image | type: opencv image
#           OR
#          image | type: Boolean, Value: False
#
#================================================================
def checkImages(path):

    #return object
    image = False

    #image types that we recognize
    valid_images = [".jpg",".gif",".png", ".jpeg"]

    #if file ends in a file extension associated with an image, we take it
    if os.path.splitext(path)[1].lower() in valid_images:
        image = cv2.imread(path, cv2.IMREAD_COLOR) 

    #return image or boolean
    return image 