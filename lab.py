#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image
import random 


def get_pixel(image, x, y):
    #store height and width as values
    height = image['height']
    width = image['width']
    #accounts for edge cases
    if y > height - 1:
        y = height - 1
    if y < 0:
        y=0
    if x > width - 1:
        x = width - 1
    if x < 0:
        x = 0 
    #calculation to turn list index into matrix index
    index = (width*y)+x
    return image['pixels'][index]

i = {'height': 3, 'width': 2, 'pixels': [0, 50, 50, 100, 100, 255]}

print(get_pixel(i, 2, 2))

def set_pixel(image, x, y, c):
    #store height and width as values
    height = image['height']
    width = image['width']
    #account for edge cases 
    if y > height - 1:
        y = height - 1
    if y < 0:
        y=0
    if x > width - 1:
        x = width - 1
    if x < 0:
        x = 0 
    #calculation to turn list index into matrix index
    index = (width*y)+x
    image['pixels'][index] = c

def apply_per_pixel(image, func):
    #initialize image made of zeros
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': len(image['pixels'])*[0],
    }
    #apply new color to each combination of pixels
    for x in range(image['width']):
        for y in range(image['height']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    #should be 256, not 255
    return apply_per_pixel(image, lambda c: 255-c)



def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Representing kernel as a dictionary
    """
    #store the width, height, and kernel dimension
    #store initialized image
    image_width = image['width']
    image_height = image['height']
    kernel_width = kernel['width']
    kernel_height = kernel['height']
    kernel_dimension = kernel['width']//2
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': len(image['pixels'])*[0]}
    
    #loop through every coord
    for x in range(image_width):
        for y in range(image_height):
            #initialize total sum
            total = 0 
            #loop through every coord in the kernel
            for y_kernel in range(kernel_width):
                for x_kernel in range(kernel_height):
                    #calculation for new x and y 
                    result_x = x-kernel_dimension+x_kernel
                    result_y = y-kernel_dimension+y_kernel
                    #add to total
                    result_val = get_pixel(image, result_x, result_y)*get_pixel(kernel, x_kernel, y_kernel)
                    total += result_val
                #set new pixel value and return new image
            set_pixel(new_image, x, y, total)
    return new_image


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    #initialize an image of all zeros
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'].copy()}
    len_pixels = len(new_image['pixels'])
    #perform calculations to round and clip image
    for i in range(len_pixels):
        new_image['pixels'][i] = int(round(new_image['pixels'][i]))
        if new_image['pixels'][i] > 255:
            new_image['pixels'][i] = 255
        if new_image['pixels'][i] < 0:
            new_image['pixels'][i] = 0 
    return new_image
        

def make_matrix(n):
    """
    Given an input n, returns a blur box for use in the Blurred Image function
    
    """
    #number that makes up the matrix depending on value of n
    num = 1/n**2
    #creates a blur box as a dictionary
    blur_box = {'height': n, 'width': n, 'pixels': [num]*n**2 }
    return blur_box 
    
def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    #access the relvant blur box and then correlates with new image
    blur_box = make_matrix(n)
    new_image = correlate(image, blur_box)
    return round_and_clip_image(new_image)


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES
    
def sharpened(image, n):
    """
    Returns a new sharpened image using relevant calculation of subtracting blurred image.
    
    This process should not mutate the input image; rather returns a new one.
    
    """
    #store image width and height
    image_width = image['width']
    image_height = image['height']
    #intialize new image and stores blurred image
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': len(image['pixels'])*[0]}
    blurred_image = blurred(image, n)
    
    #loop through every coordinate
    for x in range(image_width):
        for y in range(image_height):
            #perform calculation and set pixel
            a = 2*get_pixel(image, x, y) - get_pixel(blurred_image, x, y)
            set_pixel(new_image, x, y, a)
    return round_and_clip_image(new_image)

def edges(image):
    """
    Impletments a filter which detects the edges of an image with the Sobel Operator.
    
    Does not modify the input image; rather, it returns a new image.
    
    """
    #store image width and height and initialize new image
    image_width = image['width']
    image_height = image['height']
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': len(image['pixels'])*[0]}
    
    #sobel operator kernels
    kernel_x = {'height': 3, 'width': 3, 'pixels': [-1,0,1,-2,0,2,-1,0,1]}
    kernel_y = {'height': 3, 'width': 3, 'pixels': [-1,-2,-1,0,0,0,1,2,1]}
    
    #creating the filters
    o_x = correlate(image, kernel_x)
    o_y = correlate(image, kernel_y)

    #perform relvant calculation for each pixel 
    for x in range(image_width):
        for y in range(image_height):
            a = ((get_pixel(o_x, x, y))**2+(get_pixel(o_y, x, y))**2)**0.5
            set_pixel(new_image, x, y, a)
    return round_and_clip_image(new_image)


def merge(list1, list2, list3): 
      
    merged_list = [(list1[i], list2[i], list3[i]) for i in range(0, len(list1))] 
    return merged_list 


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    #splits image into three greyscale versions
    def split_grayscale(colorimage):
        image1 = {'height': colorimage['height'], 'width': colorimage['width'], 
              'pixels': [x[0] for x in colorimage['pixels']]}
        image2 = {'height': colorimage['height'], 'width': colorimage['width'], 
              'pixels': [x[1] for x in colorimage['pixels']]}
        image3 = {'height': colorimage['height'], 'width': colorimage['width'], 
              'pixels': [x[2] for x in colorimage['pixels']]}
        #applies filter to each image
        l1 = filt(image1)
        l2 = filt(image2)
        l3 = filt(image3)
        #returns new image
        new_image = {'height': colorimage['height'], 'width': colorimage['width'], 'pixels': merge(l1['pixels'],l2['pixels'],l3['pixels'])}
        return new_image
    #returns function
    return split_grayscale
          
         

def make_blur_filter(n):
    #uses blur filter from before and returns a function
    def blur_image(image):
        blur_box = make_matrix(n)
        new_image = correlate(image, blur_box)
        return round_and_clip_image(new_image)
    return blur_image


def make_sharpen_filter(n):
    #uses sharpen filter from before and returns a function
    def sharpen_image(image):
        #store image width and height
        image_width = image['width']
        image_height = image['height']
        #intialize new image and stores blurred image
        new_image = {'height': image['height'], 'width': image['width'], 'pixels': len(image['pixels'])*[0]}
        blurred_image = blurred(image, n)
    
        #loop through every coordinate
        for x in range(image_width):
            for y in range(image_height):
            #perform calculation and set pixel
                a = 2*get_pixel(image, x, y) - get_pixel(blurred_image, x, y)
                set_pixel(new_image, x, y, a)
        print(new_image)
        return round_and_clip_image(new_image)
    return sharpen_image


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def filter_image(image):
        #stores initial value
        initial_val = image
        #applies filters in a loop
        for i in range(len(filters)): 
            #changes initial value
            initial_val = filters[i](initial_val)
            #returns value
        return initial_val
    return filter_image
        
            
# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    
    """
    #initializes new image
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'].copy()}
    #applies each of the functions in iterative order
    for i in range(ncols):
        greyscale = greyscale_image_from_color_image(new_image)
        energy_map = compute_energy(greyscale)
        cme = cumulative_energy_map(energy_map)
        seam = minimum_energy_seam(cme)
        new_image = image_without_seam(new_image, seam)
    #returns new image
    return new_image 


# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    #inializes values
    pixels = image['pixels']
    new_pixels = []
    #calculates greyscale value for each tuple
    for tup in pixels:
        v = round(.299*tup[0]+.587*tup[1]+.114*tup[2])
        new_pixels.append(v)
    #returns new greyscale value
    grey = {'height': image['height'], 'width': image['width'], 
            'pixels': new_pixels}
    return grey
        

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    #applies edges to grey
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    #initializes values
    image_width = energy['width']
    image_height = energy['height']
    new_image = {'height': energy['height'], 'width': energy['width'], 'pixels': len(energy['pixels'])*[0]}
    
    #loops through first row and sets values
    for x in range(image_width):
        a = get_pixel(energy,x,0)
        set_pixel(new_image, x, 0, a)
    #loops through next row and sets values, getting the minimum of adjacent ones
    for y in range(1, image_height):
        for x in range(image_width):
            a = get_pixel(energy,x,y)
            b = get_pixel(new_image, x-1, y-1)
            c = get_pixel(new_image, x, y-1)
            d = get_pixel(new_image, x+1,y-1)
            #sets pixel value
            set_pixel(new_image, x, y, a+min(b,c,d))
        #returns new image
    return new_image



def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    print('start')
    #initializes values
    image_width = cem['width']
    image_height = cem['height']
    #y-value always is height minus one
    y_val = image_height-1
    indices = []
    find_min = []
    
    #find minimum value in the bottom row 
    for x in range(image_width):
        a = get_pixel(cem,x,y_val)
        find_min.append(a)
    min_val = min(find_min)
    x_val = find_min.index(min_val)
    indices.append((image_width*y_val)+x_val)
    #loop through the y values backwards 
    for y_val in reversed(range(image_height-1)):
        #calculate the left bound and the right bound
        left_bound = max(0, x_val-1)
        right_bound =min(image_width-1, x_val+1)
        #set x values and y values accordingly 
        x_vals = range(left_bound, right_bound+1)
        y_vals = [get_pixel(cem, x, y_val) for x in x_vals]
        #find the minimum value
        min_index = y_vals.index(min(y_vals))
        x_val = x_vals[min_index]
        #append the values to the indices
        indices.append((image_width*y_val)+x_val)
        #return the indices
    return indices
    


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    #initialize nw image
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'].copy()}
    #loop through seam
    for i in range(len(seam)):
        #delete the seam values from the image
        del new_image['pixels'][seam[i]]
    #return new image
    new_image = {'height': image['height'], 'width': image['width']-1, 'pixels': new_image['pixels']}
    return new_image
    


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()
    
def shaded_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    #inializes values
    pixels = image['pixels']
    new_pixels = []
    #calculates greyscale value for each tuple
    for tup in pixels:
        a = int(.5*tup[0])
        b = int(.5*tup[1])
        c = int(.5*tup[2])
        new_pixels.append((a, b, c))
    #returns new greyscale value
    grey = {'height': image['height'], 'width': image['width'], 
            'pixels': new_pixels}
    return grey
    


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
#    cat = load_color_image('test_images/cat.png')
#    c_filt = color_filter_from_greyscale_filter(inverted)
#    new_image = c_filt(cat)
#    save_color_image(new_image, 'test_cat.png')
    
#    python = load_color_image('test_images/python.png')
#    blurred_filter = make_blur_filter(9)
#    final_python = color_filter_from_greyscale_filter(blurred_filter)(python)
#    save_color_image(final_python, 'test_python.png')
    
#    sparrowchick = load_color_image('test_images/sparrowchick.png')
#    sharpened_filter = make_sharpen_filter(7)
#    final_sparrow = color_filter_from_greyscale_filter(sharpened_filter)(sparrowchick)
#    save_color_image(final_sparrow, 'test_sparrow.png')
    
#    frog = load_color_image('test_images/frog.png')
#    filter1 = color_filter_from_greyscale_filter(edges)
#    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
#    filt = filter_cascade([filter1, filter1, filter2, filter1])
#    final_frog = filt(frog)
#    #final_frog=filter1(filter2(filter1(filter1(frog))))
#    save_color_image(final_frog, 'test_frog.png')
    
    cat = load_color_image('test_images/twocats.png')
    new_cat = shaded_image_from_color_image(cat)
    save_color_image(new_cat, 'test_cat.png')
    
    
  