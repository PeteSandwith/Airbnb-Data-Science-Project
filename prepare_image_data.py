
import cv2 as cv
import pandas as pd
import os
# Function has to :
# 1. Load an image from Airbnb-Data-Science-Project/airbnb_property_listings/images 
# 2. Resizes it to the same height and width; set the height of the smallest image as the height for all of the other images.
#    Your code should maintain the aspect ratio of the image and adjust the width proportionally to the change in height, rather than just squashing it vertically.
# 3. Check that the image is in RBG format; if it isn't it should be discarded.
# 4. Save the processed image in the folder Airbnb-Data-Science-Project/airbnb_property_listings/processed_images

def process_images():
    dataframe = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')
    uuids = dataframe['ID'].values
    for id in uuids:
        letters = ['a', 'b', 'c', 'd', 'e']
        for letter in letters:
            image = load_image(id, letter)
            if image is None:
                break
            if letter == 'a':
                # Creates a directory inside processed_images where all images of the same uuid will be saved. 
                # Only does this once for each uuid and only does it AFTER checking that load_image has not returned a NoneType object.
                os.mkdir('airbnb-property-listings/processed_images/{}'.format(id))
            image = resize_image(image)
            save_image(image, id, letter)


# The following function loads an image, based on the uuid of the airbnb item and a letter from a to e
def load_image(id, letter):   
    unprocessed_image_location = 'airbnb-property-listings/images/{}'.format(id) + '/{}'.format(id) + '-{}.png'.format(letter)
    image = cv.imread(unprocessed_image_location, cv.IMREAD_COLOR)
    return image

#The following function resizes an image. It returns an image with a height of 400 and the same proportions as the original image.
def resize_image(image):
    dimensions = image.shape
    scale = 400 / dimensions[0]
    image = cv.resize(image, (0,0), fx = scale, fy = scale)
    return image

#The following function saves an image in the new processed images folder
def save_image(image, id, letter):
    processed_image_location = 'airbnb-property-listings/processed_images/{}'.format(id) + '/{}'.format(id) + '-{}.png'.format(letter)
    print(processed_image_location)
    Save = cv.imwrite(processed_image_location, image)


if __name__ == "__main__":
    process_images()

