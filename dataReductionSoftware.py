# Library with the spectral data class to be used in the data
# reduction for the CHEEESE experiment

import numpy as np
import scipy as sp 
from scipy import optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


def polyfit2d(x, y, z, kx=3, ky=3, order=None): 
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))
    
    
    From stack overflow: https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

class LineProps():

    def __init__(self, lineName, lineWavelength, lineWidth, lineShift):
        self.lineList = lineList
        self.lineDict = lineDict
        self.rawSpectrum = rawSpectrum
        self.rawSpectrumLimits = rawSpectrumLimits


class spectrum():

    def __init__(self, rawSpectrum, rawSpectrumLimits, plateScale=3.43):
        self.rawSpectrum = rawSpectrum
        self.rawSpectrumLimits = rawSpectrumLimits
        self.numSpectrumRows = rawSpectrumLimits[1] - rawSpectrumLimits[0]
        self.rawSpectrumNX = rawSpectrum.shape[0]
        self.rawSpectrumNY = rawSpectrum.shape[1]
        self.plateScale = plateScale
    
    def parabola(self, x, a, b, c):
        return a*x**2 + b*x + c

    def gaussian(self, x, mu, sigma, A = 1):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))

    def fit_parabola(self, x, y):
        popt, pcov = opt.curve_fit(self.parabola, x, y)
        return popt

    def fit_gaussian(self, x, y):
        popt, pcov = opt.curve_fit(self.gaussian, x, y)
        return popt
    
    def line_fit_science(self, lineList, waveLim, numPoints=5):
        """
        Fit the lines in the spectrum. The lines are fit by fitting a Gaussian to the line core and
        the center of the line is recorded for each line. The line center is then used to compute the
        dispersion coefficients.

        Input:
            -- lineList: dictionary with the line name as the key and the line wavelength as the value
            -- waveLim: limits of the wavelength range to fit the lines
            -- numPoints: number of points to fit the line core; default is 5
        """
        lineCenter = []

        for el in self.rawSpectrum:
            minLocation = np.argmax(el[waveLim[0]:waveLim[1]]) + waveLim[0]
            x = np.linspace(minLocation-numPoints/2, minLocation+numPoints/2,
                            num=numPoints)
            params = self.fit_gaussian(x, el[(minLocation-numPoints//2):(minLocation+numPoints//2+1)])

            # params are [LineCenter, Width, Amplitude]

            lineCenter.append(params)

        return np.array(lineCenter)

    def line_fit(self, array, waveLim, numPoints=5):
        """
        Fit the lines in the spectrum. The lines are fit by fitting a parabola to the line core and 
        the center of the line is recorded for each line. The line center is then used to compute the
        dispersion coefficients. 

        Input:
            -- array: 2D array with the spectrum
            -- waveLim: limits of the wavelength range to fit the lines
            -- numPoints: number of points to fit the line core; default is 5

        """
        lineCenter = []
        # minLocation = np.argmin(array[0, :])
        for el in array:
            # minLocation = np.argmin(el[(minLocation-numPoints//2):(minLocation+numPoints//2+1)])
            minLocation = np.argmin(el[waveLim[0]:waveLim[1]]) + waveLim[0]
            x = np.linspace(minLocation-numPoints/2, minLocation+numPoints/2,
                            num=numPoints)
            
            params = self.fit_parabola(x, el[(minLocation-numPoints//2):(minLocation+numPoints//2+1)])
            lineCentertmp = -1* params[1]/(2*params[0])

            if (lineCentertmp < waveLim[0]):
                lineCentertmp = waveLim[0]
            elif (lineCentertmp > waveLim[1]):
                lineCentertmp = waveLim[1]

            lineCenter.append(lineCentertmp)
            
        return np.array(lineCenter)

    def compute_lines(self, lineList):
        """
        Compute the spectral line location from a list of lines. The list of lines 
        is a dictionary with the line name as the key and the line wavelength as the value.

        Input:
            -- lineList: dictionary with the line name as the key and the line wavelength as the value
        """
        self.lineDict = {} # dictionary containing the locations of lines in 
        for el in lineList:
            print(f"Fitting {el} Å line")
            self.lineDict[el] = self.line_fit(self.rawSpectrum[self.rawSpectrumLimits[0]:self.rawSpectrumLimits[1], :],
                                         lineList[el])
        self.lineList = np.array([el for el in self.lineDict.keys()]).astype(int)
    
    def compute_dispersion(self, smoothLineFits=True,
                           smoothLineFitsSigma=10):
        """
        Compute the dispersion coefficients for the spectrum. The dispersion coefficients
        are saved in the dispersionCoeffs attribute of the class. The dispersion coefficients
        are computed by fitting a parabola to the lines in the spectrum. The dispersion coefficients
        are then used to compute the dispersion grid. The dispersion grid is saved in the dispersionGrid
        attribute of the class. The dispersion grid is computed by evaluating the wavelength (pixel locations)
        of the know IR spectral features. The dispersion grid is then used to interpolate the raw
        spectrum onto a uniform grid in wavelength. 

        The uniform grid is saved in the lambdaUniform attribute
        
        Input:
            -- smoothLineFits: if True, the line fits are smoothed by convolving with a Gaussian filter;
                                default is True
            -- smoothLineFitsSigma: sigma of the Gaussian filter used to smooth the line fits; default is 10

        """
        
        if smoothLineFits == True:
            filterRange = np.linspace(0, smoothLineFitsSigma*4-1, num=smoothLineFitsSigma*4)
            self.filter = self.gaussian(filterRange, smoothLineFitsSigma*4//2, smoothLineFitsSigma)

            for el in self.lineList:
                self.lineDict[el] = np.convolve(self.lineDict[el], self.filter, mode="Same")
        self.lineFitArray = np.array([self.lineDict[el] for el in self.lineDict.keys()]).astype(int).T
        
        self.dispersionCoeffs = [self.fit_parabola(el[1], el[0]) 
                                for el in zip([self.lineList] * int(self.lineFitArray.shape[0]),
                                              self.lineFitArray)]
        self.dispersionCoeffs = np.array(self.dispersionCoeffs)
        
        # smooth dispersion Coefficients
        print(f"Shape of dispCoeff: {self.dispersionCoeffs.shape}")

        self.dispersionGrid = [self.parabola(np.linspace(0, self.rawSpectrumNY-1, num=self.rawSpectrumNY),
                               el[0], el[1], el[2]) for el in self.dispersionCoeffs]
        self.dispersionGrid = np.array(self.dispersionGrid)
        # smooth by convolving with a 2D array 

    def saveDispersionGrid(self, outputName="dispersion_file.npz"):
        """
        Save the dispersion grid to a file. The file will be a .npz file
        
        Input:
            -- outputName: name of the output file
        """
        np.savez(outputName, 
                 dispersionGrid = self.dispersionGrid,
                 dispersionCoeffs = self.dispersionCoeffs)
        
    def loadDispersionGrid(self, inputName="/Users/mmolnar/Work/Hanle_CME_Bz/CHEESE_data/24_04_08_Eclipse_data/dispersion_file.npz"):
        ''' 
        Load the dispersion grid from a file. The file should be a .npz file
        
        Input: 
            - inputName: name of the input file; default is dispersion_file.npz
        '''

        with np.load(inputName) as spFile:
            self.dispersionGrid = spFile["dispersionGrid"]
            self.dispersionCoeffs = spFile["dispersionCoeffs"]


    def recomputeUniformGrid(self, minLambda = 10675, 
                             maxLambda = 11100, R = 11000):
        """
        Recompute the uniform grid in wavelength. The uniform grid is computed by evaluating the
        wavelength of the known IR spectral features. The dispersion grid is used to interpolate the
        raw spectrum onto a uniform grid in wavelength. The uniform grid is saved in the lambdaUniform
        attribute of the class. The interpolated spectrum is saved in the spectrumUniform attribute of the class.

        """

        self.lambdaUniform = np.linspace(minLambda, maxLambda, 
                                         num=1180)
        # for el in zip(self.dispersionGrid, 
        #                                       self.rawSpectrum[self.rawSpectrumLimits[0]:self.rawSpectrumLimits[1],
        #                                                       :]):
        #     # print(el[0].shape, el[1].shape)
        # # print(self.lambdaUniform.shape)
        print(f"recomputing Uniform grid")
        print(f"{self.rawSpectrumLimits[0], self.rawSpectrumLimits[1]}")
        self.spectrumUniform = np.array([np.interp(self.lambdaUniform, np.flip(el[0]), np.flip(el[1])) 
                                         for el in zip(self.dispersionGrid, 
                                              self.rawSpectrum[self.rawSpectrumLimits[0]:self.rawSpectrumLimits[1],
                                                              :])])