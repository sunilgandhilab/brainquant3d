# cython: infer_types=True
# cython: language_level=3
import cv2
import numpy as np
cimport cython
from libc.math cimport sqrt, round


"""
Fully Ported to Python from ImageJ's Background Subtractor.
Only works for 8-bit greyscale images currently.
Based on the concept of the rolling ball algorithm described
in Stanley Sternberg's article,
"Biomedical Image Processing", IEEE Computer, January 1983.
Imagine that the 2D grayscale image has a third (height) dimension by the image
value at every point in the image, creating a surface. A ball of given radius
is rolled over the bottom side of this surface; the hull of the volume
reachable by the ball is the background.
http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html

Based on Maksym Balatsko pure Python implimentation:
(https://github.com/mbalatsko/opencv-rolling-ball/blob/master/cv2_rolling_ball/background_subtractor.py_.
"""

#int8 use unsigned char
ctypedef fused DTYPE:
    unsigned char
    unsigned short

cdef float NEG_INFINITY = float("-inf")
cdef float INFINITY     = float("inf")


cdef int X_DIRECTION = 0
cdef int Y_DIRECTION = 1
cdef int DIAGONAL_1A = 2
cdef int DIAGONAL_1B = 3
cdef int DIAGONAL_2A = 4
cdef int DIAGONAL_2B = 5

cdef class BackgroundSubtract:

    cdef Py_ssize_t width
    cdef Py_ssize_t height

    cdef Py_ssize_t s_width
    cdef Py_ssize_t s_height

    cdef int shrink_factor

    cdef Py_ssize_t ball_width
    cdef list ball_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef rolling_ball_background(self, DTYPE[:,:] img, int radius):
        """
        Calculates and subtracts or creates background from image.
        Parameters
        ----------
        img : uint8 np array
            Image
        radius : int
            Radius of the rolling ball creating the background (actually a
                          paraboloid of rotation with the same curvature)
        Returns
        -------
        img, background : uint8 np array
        Background subtracted image, Background

        """
        # setup data
        img_arr = np.array(img)

        self.height     = img_arr.shape[0]
        self.width      = img_arr.shape[1]
        self.s_height   = img_arr.shape[0]
        self.s_width    = img_arr.shape[1]

        cdef DTYPE[:]   img_flat   = img_arr.reshape(self.height * self.width)
        cdef float[:]   float_img  = img_arr.reshape(self.height * self.width).astype('float32')

        # setup subtract
        self._create_ball(radius)
        
        cdef bint shrink
        shrink = self.shrink_factor > 1
        if shrink:
            small_img = self._shrink_image(float_img, self.shrink_factor)
        else:
            small_img = float_img
        
        # subtract
        small_img = self._roll_ball(small_img)

        if shrink:
            float_img = self._enlarge_image(small_img, float_img)

        cdef float offset = 0.5
        cdef DTYPE max_val = -1 # set to max value
        cdef long value

        for p in range(0, self.width*self.height):
            value = int(float_img[p] + offset)
            value = img_flat[p] - value
            value = max((value, 0)) # ensure in range
            value = min((value, max_val)) # ensure in range
            img[p / self.width, p % self.width] = value

        background = np.array(float_img).reshape((self.height, self.width)).astype(img_arr.dtype)

        return np.array(img), background

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float[:] _roll_ball(self, float[:] float_img):
        """
        Rolls a filtering object over an image in order to find the
        image's smooth continuous background.  For the purpose of explaining this
        algorithm, imagine that the 2D grayscale image has a third (height)
        dimension defined by the intensity value at every point in the image.  The
        center of the filtering object, a patch from the top of a sphere having
        radius 'radius', is moved along each scan line of the image so that the
        patch is tangent to the image at one or more points with every other point
        on the patch below the corresponding (x,y) point of the image.  Any point
        either on or below the patch during this process is considered part of the
        background.
        """

        cdef Py_ssize_t height      = self.s_height
        cdef Py_ssize_t width       = self.s_width
        cdef Py_ssize_t ball_width  = self.ball_width
        cdef int        radius      = ball_width / 2

        # cache will span multi lines
        cdef float[:] cache = np.zeros((width * ball_width), dtype = 'float32')
        cdef float[:] z_ball = np.array(self.ball_data, dtype = 'float32')

        # loop variables
        cdef float z, z_reduced, z_min
        cdef int next_line_to_write, next_line_to_read, src, dest, x0, y0, bp, y_ball0, x_ball0
        cdef Py_ssize_t x, y, yp, xp, y_end, x_end 

        for y in range(-radius, height + radius): # whole image
            next_line_to_write = (y + radius) % ball_width
            next_line_to_read = y + radius
            if next_line_to_read < height: # as long as in bounds
                src = next_line_to_read * width
                dest = next_line_to_write * width
                cache[dest:dest+width] = float_img[src:src+width]
                float_img[src:src+width] = NEG_INFINITY # to mark as complete?

            y0 = max((0, y - radius))
            y_ball0 = y0 - y + radius
            y_end = y + radius
            if y_end >= height:
                y_end = height - 1
            for x in range(-radius, width + radius): # whole image
                z = INFINITY
                x0 = max((0, x - radius))
                x_ball0 = x0 - x + radius # always = 0?
                x_end = x + radius
                if x_end >= width:
                    x_end = width - 1

                y_ball = y_ball0
                for yp in range(y0, y_end + 1):
                    cache_pointer = (yp % ball_width) * width + x0
                    bp = x_ball0 + y_ball * ball_width
                    for xp in range(x0, x_end + 1):
                        z_reduced = cache[cache_pointer] - z_ball[bp]
                        if z > z_reduced:
                            z = z_reduced
                        cache_pointer += 1
                        bp += 1
                    y_ball += 1

                y_ball = y_ball0
                for yp in range(y0, y_end + 1):
                    p = x0 + yp * width
                    bp = x_ball0 + y_ball * ball_width
                    for xp in range(x0, x_end + 1):
                        z_min = z + z_ball[bp]
                        if float_img[p] < z_min:
                            float_img[p] = z_min
                        p += 1
                        bp += 1
                    y_ball += 1

        return float_img

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float[:] _shrink_image(self, float[:] flat_img, int shrink_factor):
        """Shrinks image, using min value for interpolation
        """

        self.s_height = self.height / shrink_factor
        self.s_width = self.width / shrink_factor

        cdef float[:,:] img  = np.array(flat_img).reshape((self.height, self.width))
        cdef float[:] sink = np.zeros(self.s_height * self.s_width, np.float32)
        cdef float min_value

        cdef Py_ssize_t x,y, flat_idx
        for y in range(0, self.s_height):
            for x in range(0, self.s_width):
                x_mask_min = shrink_factor * x
                y_mask_min = shrink_factor * y
                min_value = _min(img[y_mask_min:y_mask_min + shrink_factor,
                            x_mask_min:x_mask_min + shrink_factor])
                flat_idx = (y * self.s_width) + x
                sink[flat_idx] = min_value

        return sink

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float[:] _enlarge_image(self, float[:] small_img, float[:] float_img):
        """Enlarges image. assume image is flattened.
        """
        height, width = self.height, self.width
        s_height, s_width = self.s_height, self.s_width

        cdef int[:]   x_s_indices, y_s_indices
        cdef float[:] x_weigths, y_weights
        x_s_indices, x_weigths = self._make_interpolation_arrays(width, s_width)
        y_s_indices, y_weights = self._make_interpolation_arrays(height, s_height)


        cdef float[:] line0 = np.zeros((width), dtype = 'float32')
        cdef float[:] line1 = np.zeros((width), dtype = 'float32')

        cdef int y_s_line0
        cdef float x_val, weight
        cdef Py_ssize_t x,y, s_y_ptr


        for x in range(0, width):
            x_val = small_img[x_s_indices[x]] * x_weigths[x] + small_img[x_s_indices[x] + 1] * (1.0 - x_weigths[x])
            line1[x] = x_val

        y_s_line0 = -1
        for y in range(0, height):
            if y_s_line0 < y_s_indices[y]:
                line0, line1 = line1, line0
                y_s_line0 += 1
                s_y_ptr = (y_s_indices[y] + 1) * s_width
                for x in range(0, width):
                    line1[x] = small_img[s_y_ptr + x_s_indices[x]] * x_weigths[x] + small_img[s_y_ptr + x_s_indices[x] + 1] * (1.0 - x_weigths[x])
            weight = y_weights[y]
            p = y * width
            for x in range(0, width):
                float_img[p] = line0[x] * weight + line1[x] * (1.0 - weight)

                p += 1
        return float_img

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef tuple _make_interpolation_arrays(self, Py_ssize_t length, int s_length):

        cdef int[:] s_indices = np.zeros((length), dtype = np.intc)
        cdef float[:] weights = np.zeros((length), dtype = 'float32')

        cdef Py_ssize_t i, s_idx
        cdef float distance

        for i in range(0, length):
            s_idx = (i - self.shrink_factor / 2) / self.shrink_factor
            if s_idx >= s_length - 1:
                s_idx = s_length - 2
            s_indices[i] = s_idx
            distance = (i + 0.5) / self.shrink_factor - (s_idx + 0.5)
            weights[i] = 1.0 - distance
        return s_indices, weights

    @cython.cdivision(True)
    cdef void _create_ball(self, int radius):
        """
            A rolling ball (or actually a square part thereof)
            Here it is also determined whether to shrink the image
        """

        if radius <= 10:
            self.shrink_factor = 1
            arc_trim_per = 24
        elif radius <= 30:
            self.shrink_factor = 2
            arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            arc_trim_per = 32
        else:
            self.shrink_factor = 8
            arc_trim_per = 40

        cdef int small_ball_radius = radius / self.shrink_factor

        if small_ball_radius < 1:
            small_ball_radius = 1


        cdef int x_val, y_val
        cdef float temp

        cdef int r_square = small_ball_radius * small_ball_radius
        cdef int x_trim = arc_trim_per * small_ball_radius / 100
        cdef int half_width = int(round(small_ball_radius - x_trim))
        self.ball_width = 2 * half_width + 1
        self.ball_data = [0] * (self.ball_width * self.ball_width)

        cdef Py_ssize_t x, y, p = 0
        for y in range(self.ball_width):
            for x in range(self.ball_width):
                x_val = x - half_width
                y_val = y - half_width

                temp = r_square - x_val * x_val - y_val * y_val
                self.ball_data[p] = sqrt(temp) if temp > 0 else 0

                p += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _min(float[:,:] arr):

    cdef Py_ssize_t x_max = arr.shape[0]
    cdef Py_ssize_t y_max = arr.shape[1]
    cdef Py_ssize_t x, y

    cdef float val = arr[0,0]
    for y in range(y_max):
        for x in range(x_max):
            if arr[x,y] < val:
                val = arr[x,y]

    return val

def subtract_background_rolling_ball(DTYPE[:,:] img, int radius):
    """Subtracts background via subtracting a morphological opening from the original image

    Attributes:
        img (array): uint8 or uint16 nnumpy array. must be little
        radius (int): Radius of the rolling ball creating the background (actually a
                      paraboloid of rotation with the same curvature)

        size (tuple): Size for the structure element of the morphological opening.
    Returns:
        img, background: Background subtracted image, Background
    """
    bs = BackgroundSubtract()
    return bs.rolling_ball_background(img, radius)