# Map-Segmentation-by-Color
This project segments an input image of the United States of America into it's individual states based on their color. Once the app opens, clicking on a state will cause the state to be removed from the original image, enlarged, and then superimposed on the original image with a lowered opacity.

# Notes
Methods used are:
  - k-means clustering for state pixel segmentation
  - masking to get all pixels of matching colors
  - dilation to enlarge all selected states in the mask
  - flood filling from the selected region to ensure collection of entire disonnected state
  - masking of flood fill onto original mask
  - masking new mask onto original image
  - enlarging and superimposing mask onto original image

## Usage:
<pre>
map_segment [-h] -t input_image 
    -t : Target Image (t)
            </pre>
