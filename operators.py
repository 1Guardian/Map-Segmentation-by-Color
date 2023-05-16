from imports import *

#================================================================
#
# Function: opCOPY(primary)
#
# Description: copies an image via a deep copy
#
# Returns: image | openCV image
#
#================================================================
def opCOPY(primary):

    return np.array(primary, copy=True)  

#================================================================
#
# Function: opOR(primary, secondary)
#
# Description: logical ORs two images
#
# Returns: image | openCV image
#
#================================================================
def opOR(primary, secondary):

    #determine if we have to do a layer-wise OR
    primarylayercount = 1
    secondarylayercount = 1
    if len(primary.shape) > 2:
        primarylayercount = primary.shape[2]
    if len(secondary.shape) > 2:
        secondarylayercount = secondary.shape[2]

    #copy image
    retimg = opCOPY(primary)

    #OR each layer
    if (primarylayercount != 1 and secondarylayercount == 1):
        for i in range(primarylayercount):
            retimg[:,:,i] = retimg[:,:,i] | secondary
    elif (primarylayercount != 1 and secondarylayercount != 1):
        for i in range(primarylayercount):
            retimg[:,:,i] = retimg[:,:,i] | secondary[:,:,i]
    else:
        retimg = retimg | secondary

    #OR the image 
    return retimg
    
#================================================================
#
# Function: opAND(primary, secondary)
#
# Description: logical ANDs two images
#
# Returns: image | openCV image
#
#================================================================
def opAND(primary, secondary):

    #determine if we have to do a layer-wise OR
    primarylayercount = 1
    secondarylayercount = 1
    if len(primary.shape) > 2:
        primarylayercount = primary.shape[2]
    if len(secondary.shape) > 2:
        secondarylayercount = secondary.shape[2]

    #copy image
    retimg = opCOPY(primary) 

    #AND each layer
    if (primarylayercount != 1 and secondarylayercount == 1):
        for i in range(primarylayercount):
            retimg[:,:,i] = retimg[:,:,i] & secondary
    elif (primarylayercount != 1 and secondarylayercount != 1):
        for i in range(primarylayercount):
            retimg[:,:,i] = retimg[:,:,i] & secondary[:,:,i]
    else:
        retimg = retimg & secondary
    return retimg

#================================================================
#
# Function: opNOT(primary)
#
# Description: logical NOTs an image
#
# Returns: image | openCV image
#
#================================================================
def opNOT(primary):

    #get height and width and make copy
    retimg = opCOPY(primary)

    #NOT the image 
    retimg[retimg < 1] = 2
    retimg[retimg > 2] = 0
    retimg[retimg > 0] = 255
    return retimg

#================================================================
#
# Function: opOVER(primary, secondary, maskone, masktwo)
#
# Description: applies the OVER operator to two images
#
# Returns: image | openCV image
#
#================================================================
def opOVER(primary, secondary, maskone, masktwo):

    #I1 AND M1
    LHS = opAND(primary, maskone)

    #I2 AND M2 AND NOT(M1)
    RHS = opAND((opAND(secondary, masktwo)), opNOT(maskone))

    #OR the two together
    Ir = opOR(LHS, RHS)

    #get maskr
    maskr = opOR(maskone, masktwo)

    #and the mask and the image
    return opAND(Ir, maskr)

#================================================================
#
# Function: opIN(primary, secondary, maskone, masktwo)
#
# Description: Applies Porter-Duff IN operator to two 
#              images and their masks
#
# Returns: image | openCV image
#
#================================================================
def opIN(primary, secondary, maskone, masktwo):

    #OR the two together
    Ir = opCOPY(primary)

    #get maskr
    maskr = opAND(maskone, masktwo)

    #and the mask and the image
    return opAND(Ir, maskr)

#================================================================
#
# Function: opIN(primary, secondary, maskone, masktwo)
#
# Description: Applies Porter-Duff OUT operator to two 
#              images and their masks
#
# Returns: image | openCV image
#
#================================================================
def opOUT(primary, secondary, maskone, masktwo):

    #OR the two together
    Ir = opCOPY(primary)

    #get maskr
    maskr = opAND(maskone, opNOT(masktwo))

    #and the mask and the image
    return opAND(Ir, maskr)

#================================================================
#
# Function: opIN(primary, secondary, maskone, masktwo)
#
# Description: Applies Porter-Duff ATOP operator to two 
#              images and their masks
#
# Returns: image | openCV image
#
#================================================================
def opATOP(primary, secondary, maskone, masktwo):

    #I1 AND M1
    LHS = opAND(primary, maskone)

    #I2 AND NOT(M2)
    RHS = opAND(secondary, opNOT(masktwo))

    #OR the two together
    Ir = opOR(LHS, RHS)

    #get maskr
    maskr = opCOPY(masktwo)

    #and the mask and the image
    return opAND(Ir, maskr)

#================================================================
#
# Function: opIN(primary, secondary, maskone, masktwo)
#
# Description: Applies Porter-Duff XOR operator to two 
#              images and their masks
#
# Returns: image | openCV image
#
#================================================================
def opXOR(primary, secondary, maskone, masktwo):

    #I1 AND M1 and NOT(M2)
    LHS = opAND(opAND(primary, maskone), opNOT(masktwo))

    #I2 AND NOT(M1) AND M2
    RHS = opAND(opAND(secondary, opNOT(maskone)), masktwo)

    #OR the two together
    Ir = opOR(LHS, RHS)

    #get mask RHS and LHS
    mLHS = opAND(maskone, opNOT(masktwo))
    mRHS = opAND(opNOT(maskone), masktwo)

    #get maskr
    maskr = opOR(mLHS, mRHS)

    #and the mask and the image
    return opAND(Ir, maskr)

#================================================================
#
# Function: opClear(primary, secondary, maskone, masktwo)
#
# Description: Applies a clear operation by ANDing the image
#              with the not of its mask 
#
# Returns: image | openCV image
#
#================================================================
def opCLEAR(primary, secondary, maskone, masktwo):

    #and the mask and the image
    return opAND(primary, opNOT(maskone))