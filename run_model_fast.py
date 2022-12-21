# Run and save the output of the model BioSaVe: Large-Scale, Dynamic, Spatial, Ecohydrological
# Vegetation Model for Savanna Tree and Grass Cover and Biomass with herbivores
# copyright Leonna Szangolies, Florian Jeltsch, University of Potsdam

import numpy as np
import os
from scipy.ndimage import measurements
from RainGenerator import RainGenerator
from Landscape_herbivores import Landscape_herbivores


def run_model(settings):
            grass, trees, soil_moisture, grass_biomass, tree_biomass, fire_p, fire_distr, \
                herd, preference, amount, forage, infil, evap, poros, path, YEARS, space, s = settings
            runtime = YEARS*52
            np.random.seed(seed=s)

            # initialize rain:
            rg = RainGenerator(years=YEARS)
            rg.create_gauss_params()
            rg.let_it_rain()
            
            # initialize result arrays:
            bio_g = np.zeros((runtime, grass.shape[0], grass.shape[1]))
            bio_t = np.zeros((runtime, grass.shape[0], grass.shape[1]))
            fires = np.zeros(runtime)
            rain_out = np.zeros(runtime)
            consumption = np.zeros((runtime, grass.shape[0], grass.shape[1]))
            areas = np.array([])
            return_period = ([])
            mean_g = np.zeros(runtime)
            lower_g = np.zeros(runtime)
            higher_g = np.zeros(runtime)
            mean_bio_g = np.zeros(runtime)
            lower_bio_g = np.zeros(runtime)
            higher_bio_g = np.zeros(runtime)
            mean_t = np.zeros(runtime)
            lower_t = np.zeros(runtime)
            higher_t = np.zeros(runtime)
            mean_bio_t = np.zeros(runtime)
            lower_bio_t = np.zeros(runtime)
            higher_bio_t = np.zeros(runtime)
            pic_g = np.zeros((int(YEARS/10), grass.shape[0], grass.shape[1]))
            pic_t = np.zeros((int(YEARS/10), grass.shape[0], grass.shape[1]))
            pic_c = np.zeros((int(YEARS/10), grass.shape[0], grass.shape[1]))
            feed_size = np.zeros((runtime, len(herd)))
            all_fires = np.zeros(grass.shape)
            waiting = np.zeros((grass.shape[0], grass.shape[1]))
                
            # initialize class:
            ls = Landscape_herbivores(grass_cover=grass, tree_cover=trees, soil_moisture=soil_moisture, herd=herd,
                                      preference=preference, consumed=amount, grass_biomass=grass_biomass,
                                      tree_biomass=tree_biomass, evaporation=evap, unitary_volume_soil_porosity=poros,
                                      infiltration=infil, fire_p=fire_p, fire_distr=fire_distr, forage=forage)
            
            # weekly timestep:
            for week in range(runtime):
                rain = rg.weekly_rain[week//52, week % 52] * 0.001
                m, g, t, bio_g[week], bio_t[week], dead_grass, dead_tree, fire, herd_out, consumption[week], size = \
                    ls.weekly_timestep(rain, week)
                
                # get fire size, number and return interval:
                lw, num = measurements.label(fire, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
                fires[week] = num
                area = measurements.sum(fire, lw, index=np.arange(lw.max()+1))
                if len(area[area > 0]) > 0:
                    areas = np.append(areas, area[area > 0])
                if len((waiting*fire)[(waiting*fire) > 0]) > 0:
                    return_period = np.append(return_period, np.ndarray.flatten((waiting*fire)[(waiting*fire) > 0]))
                waiting[fire == 0] = waiting[fire == 0]+1
                waiting[fire == 1] = 0
                all_fires = all_fires+fire
                
                # save vegetation status:
                rain_out[week] = rain
                mean_g[week] = np.mean(g[(g+t) > 0])
                lower_g[week] = np.quantile(g[(g+t) > 0], 0.1)
                higher_g[week] = np.quantile(g[(g+t) > 0], 0.9)
                mean_bio_g[week] = np.mean(bio_g[week][(g+t) > 0])
                lower_bio_g[week] = np.quantile(bio_g[week][(g+t) > 0], 0.1)
                higher_bio_g[week] = np.quantile(bio_g[week][(g+t) > 0], 0.9)
                mean_t[week] = np.mean(t[(g+t) > 0])
                lower_t[week] = np.quantile(t[(g+t) > 0], 0.1)
                higher_t[week] = np.quantile(t[(g+t) > 0], 0.9)
                mean_bio_t[week] = np.mean(bio_t[week][(g+t) > 0])
                lower_bio_t[week] = np.quantile(bio_t[week][(g+t) > 0], 0.1)
                higher_bio_t[week] = np.quantile(bio_t[week][(g+t) > 0], 0.9)
                if (week % 520) == 519:
                    pic_g[week//520] = g
                    pic_t[week//520] = t
                    pic_c[week//520] = np.sum(consumption[0:week], axis=0)
                feed_size[week] = size
                
            season_bio_g = bio_g[np.arange(YEARS)*52+45]
            season_bio_t = bio_t[np.arange(YEARS)*52+45]
            
            # save data in files:
            Grass_Out = np.array([mean_g, lower_g, higher_g, grass, pic_g, np.mean(season_bio_g, axis=0), mean_bio_g,
                                  lower_bio_g, higher_bio_g], dtype='object')
            Trees_Out = np.array([mean_t, lower_t, higher_t, trees, pic_t, np.mean(season_bio_t, axis=0), mean_bio_t,
                                  lower_bio_t, higher_bio_t], dtype='object')
            Fires_Out = np.array([fires, areas, return_period, all_fires], dtype='object')
            Consumption_Out = np.array([np.sum(consumption, axis=(1, 2)), np.sum(consumption, axis=0), pic_c,
                                        feed_size], dtype='object')
            os.makedirs(path, exist_ok=True)
            np.save(path+"/Grass", Grass_Out)
            np.save(path+"/Tree", Trees_Out)
            np.save(path+"/Fire", Fires_Out)
            np.save(path+"/Consumption", Consumption_Out)
            np.save(path+"/Rain", rain_out)
