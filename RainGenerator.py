# RainGenerator Class to simulate rain timeseries
# adopted from Tietjen et al. 2010, Lohmann et al. 2012
# copyright Leonna Szangolies, Florian Jeltsch, University of Potsdam

import numpy as np

class RainGenerator:
    """
    # construct instance with standard parameters
    # get parameters for Gauss curve of mean_annual_prec    
    # get probability of rain events per day: feed amplitude, center and width into Gauss function
    # use uniform distribution to sample rain days that are also more likely than 0.05
    # -> feed these into Gauss function to gain actual rain volumes
    # sample from exponential function
    # crush daily rain into weekly patterns
    """
    def __init__(self, mean_annual_prec=300, years=1,
                 amplitude_rain=1, center_rain=1, width_rain=1,
                 amplitude_volume=1, center_volume=1, width_volume=1):
        self.mean_annual_prec = 18 + 1.24*mean_annual_prec
        self.years = years
        self.amplitude_rain = amplitude_rain
        self.center_rain = center_rain
        self.width_rain = width_rain
        self.amplitude_volume = amplitude_volume
        self.center_volume = center_volume
        self.width_volume = width_volume
        self.current_rain_series = None
        self.weekly_rain = None

    def create_gauss_params(self, relative_change=0):
        self.amplitude_rain = 0.13 + 0.00041 * self.mean_annual_prec
        self.amplitude_rain = self.amplitude_rain * (1 - relative_change
                            + 1.33 * relative_change**2
                            - (0.61 + 1.57*self.amplitude_rain) * relative_change**3)
        self.center_rain = 215
        self.width_rain = 52 + 0.007 * self.mean_annual_prec
        
        self.amplitude_volume = -14 + 4 * np.log(self.mean_annual_prec)
        self.amplitude_volume = self.amplitude_volume * (1 + relative_change)
        self.center_volume = 170
        self.width_volume = 10488
        
    def gauss(self, days, amplitude, center, width, shape=2):
        days = (days + 182 - center) % 365 + center - 182
        return amplitude * np.exp(-(days - center)**shape / (2 * width**2))
    
    def let_it_rain(self):
        days = np.tile(np.arange(0, 365), (self.years, 1))
        rain_prob_dist = self.gauss(days, self.amplitude_rain, self.center_rain, self.width_rain)
        does_it_rain = np.logical_and(np.random.uniform(size=days.shape) < rain_prob_dist,
                                      rain_prob_dist > 0.05)
        rain_vol_dist = self.gauss(days, self.amplitude_volume, self.center_volume, self.width_volume, shape=4)
        self.current_rain_series = does_it_rain * np.random.exponential(rain_vol_dist, days.shape)
        self.weekly_rain = np.add.reduceat(self.current_rain_series, np.arange(0, 364, 7), axis=1)
