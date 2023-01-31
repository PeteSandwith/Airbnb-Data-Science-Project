
import cv2 as cv
# Function has to :
# 1. Load an image from Airbnb-Data-Science-Project/airbnb_property_listings/images 
# 2. Resizes it to the same height and width; set the height of the smallest image as the height for all of the other images.
#    Your code should maintain the aspect ratio of the image and adjust the width proportionally to the change in height, rather than just squashing it vertically.
# 3. Check that the image is in RBG format; if it isn't it should be discarded.
# 4. Save the processed image in the folder Airbnb-Data-Science-Project/airbnb_property_listings/processed_images

image = cv.imread('airbnb-property-listings/images/0a26e526-1adf-4a2a-888d-a05f7f0a2f33/0a26e526-1adf-4a2a-888d-a05f7f0a2f33-a.png', cv.IMREAD_COLOR)
#cv.imshow('Image', image)
#cv.waitKey(0)
#cv.destroyAllWindows()




