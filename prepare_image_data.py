
import cv2 as cv
import pandas as pd
# Function has to :
# 1. Load an image from Airbnb-Data-Science-Project/airbnb_property_listings/images 
# 2. Resizes it to the same height and width; set the height of the smallest image as the height for all of the other images.
#    Your code should maintain the aspect ratio of the image and adjust the width proportionally to the change in height, rather than just squashing it vertically.
# 3. Check that the image is in RBG format; if it isn't it should be discarded.
# 4. Save the processed image in the folder Airbnb-Data-Science-Project/airbnb_property_listings/processed_images






id = 'f9dcbd09-32ac-41d9-a0b1-fdb2793378cf'
letters = ['a', 'b', 'c', 'd', 'e']
for letter in letters:
    unprocessed_image_location = 'airbnb-property-listings/images/{}'.format(id) + '/{}'.format(id) + '-{}.png'.format(letter)
    image = cv.imread(unprocessed_image_location, cv.IMREAD_COLOR)
    cv.imshow('Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#The following function resizes an image. It returns an image with a height of 400 and the same proportions as the original image.
def resize_image(image):
    dimensions = image.shape
    scale = 400 / dimensions[0]
    image = cv.resize(image, (0,0), fx = scale, fy = scale)
    return image








