"""
The filter provides a function to correct illumination/vignetting systematic
variations in intensity.



References:
    Fundamentals of Light Microscopy and Electronic Imaging, p. 421

"""

import numpy as np
from scipy.optimize import curve_fit

from bq3d import io
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class IlluminationCorrection(FilterBase):
    """Filter provides a function to correct systematic variations in intensity.

    The intensity image :math:`I(x)` given a flat field :math:`F(x)` and
    a background :math:`B(x)` the image is corrected to :math:`C(x)` as:

    The filter also has functionality to create flat field corections from measured
    intensity changes along a single axis, useful for lightsheet images where illumination is similar along x.
    see e.g. :func:`flatfieldLineFromRegression`.

    Call using :meth:`filter_image` with 'IlluminationCorrection' as filter.

     .. math:
         C(x) = \\frac{I(x) - B(x)}{F(x) - B(x)}

     The correction is done z slice by slice and scaled.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        model (str or array): 1D or 2D model of illumination field.
        model_axis (int): If model is 1D, the axis defines which axis it models. The model will be extended along the other axis in xy.
        background (str, array, or none): background image to be subtracted from the input image.
        scaling (str or int): scale the corrected result by this factor. Also accepts 'max' or 'mean'.

    References:
        Fundamentals of Light Microscopy and Electronic Imaging, p 421
    """

    def __init__(self):
        self.model      = None
        self.model_axis = 0
        self.background = None
        self.scaling    = 'Mean'
        super().__init__()

    def _generate_output(self):

        def _extend_model_line(model_line, model_axis, model_size):

            if len(model_line) != model_size[model_axis]:
                raise ValueError(f'1D model size of size {len(model_line)} does not match axis of size {model_size[axis]}')

            if model_axis == 0:
                # extend in y
                model_line = np.reshape(model_line, (len(model_line),1))
                model = np.tile(model_line, (1,model_size[1]))
            elif model_axis == 1:
                # extend in x
                model_line = np.reshape(model_line, (1, len(model_line)))
                model = np.tile(model_line, (model_size[0], 1))
            else:
                raise ValueError(f'extending model along axis {model_axis} not supported')

            return model

        # generate flat field model
        model = io.readPoints(self.model)
        ndims = model.shape
        if ndims == 1:
            model = _extend_model_line(model, self.model_axis, model.shape[:2])
        elif ndims > 2:
            raise ValueError(f'Illumination correction model of dimensions {ndims} not supported')

        # convert to float for scaling
        im_dtype = self.input.dtype
        model    = model.astype('float32')
        img      = self.input.astype('float32')

        # illumination correction by slice
        if self.background:
            background = io.readData(self.background).astype('float32')
            if background.shape != model.shape:
                raise ValueError(f'Background does not match model size {model.shape}')

            model = model - background
            for z in range(img.shape[0]):
                img[z] = (img[z] - background) / model
        else:
            for z in range(img.shape[0]):
                img[z] = img[z] / model

        # rescale image
        if isinstance(self.scaling, str):
            if self.scaling.lower() == "mean":
                # scale back by average flat field correction:
                factor = self.input.mean()
            elif self.scaling.lower() == "max":
                factor = self.input.max()
            else:
                raise RuntimeError(f'Scaling method {scaling} not recognized')
        else:
            factor = self.scaling

        return (img * factor).astype(im_dtype)

    def flatfieldLineFromRegression(self, data, method='polynomial', reverse=None,):
        """Create flat field line fit from a list of positions and intensities
        The fit is either to be assumed to be a Gaussian:

        .. math:
            I(x) = a \\exp^{- (x- x_0)^2 / (2 \\sigma)) + b"

        or follows a order 6 radial polynomial

        .. math:
            I(x) = a + b (x- x_0)^2 + c (x- x_0)^4 + d (x- x_0)^6

        Arguments:
            data (array): intensity data as vector of intensities or (n,2) dim array of positions d=0 and intensities measurements d=1:-1
            method (str): method to fit intensity data, 'gaussian' or 'polynomial'
            reverse (bool): reverse the line fit after fitting

        Returns:
            array: fitted intensities on points
        """

        data = io.readPoints(data)

        # split data
        if len(data.shape) == 1:
            x = np.arange(0, data.shape[0])
            y = data
        elif len(data.shape) == 2:
            x = data[:, 0]
            y = data[:, 1:-1]
        else:
            raise RuntimeError('flatfieldLineFromRegression: input data not a line or array of x,i data')

        # calculate mean of the intensity measurements
        ym = np.mean(y, axis=1)

        if method.lower() == 'polynomial':
            ## fit r^6
            mean = sum(ym * x) / sum(ym)

            def f(x, m, a, b, c, d):
                return a + b * (x - m) ** 2 + c * (x - m) ** 4 + d * (x - m) ** 6

            popt, pcov = curve_fit(f, x, ym, p0=(mean, 1, 1, 1, .1))
            m = popt[0];
            a = popt[1];
            b = popt[2]
            c = popt[3];
            d = popt[4]

            self.log.debug("polynomial fit: %f + %f (x- %f)^2 + %f (x- %f)^4 + %f (x- %f)^6" % (a, b, m, c, m, d, m))

            def fopt(x):
                return f(x, m=m, a=a, b=b, c=c, d=d)

            flt = list(map(fopt, list(range(0, int(x[-1])))))

        elif method.lower() == 'gaussian':
            ## Gaussian fit

            mean = sum(ym * x) / sum(ym)
            sigma = sum(ym * (x - mean) ** 2) / (sum(ym))

            def f(x, a, m, s, b):
                return a * np.exp(- (x - m) ** 2 / 2 / s) + b

            popt, pcov = curve_fit(f, x, ym, p0=(1000, mean, sigma, 400))
            a = popt[0];
            m = popt[1];
            s = popt[2];
            b = popt[3]

            self.log.debug("Gaussian fit: %f exp(- (x- %f)^2 / (2 %f)) + %f" % (a, m, s, b))

            def fopt(x):
                return f(x, a=a, m=m, s=s, b=b)

        else:
            raise ValueError(f'method {method} not recognized.')

        if reverse:
            flt.reverse()

        return flt

filter_manager.add_filter(IlluminationCorrection())

