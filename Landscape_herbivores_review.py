# BioSaVe: Large-Scale, Dynamic, Spatial, Ecohydrological Vegetation Model for Savanna Tree and Grass Cover and Biomass
# with herbivores
# copyright Leonna Szangolies, Florian Jeltsch, University of Potsdam

import numpy as np
from scipy.ndimage import measurements
from scipy.sparse import diags
import scipy.stats as stats
import math
from RainGenerator import RainGenerator


class Landscape_herbivores:
    """
    Constructs and handles the landscape.
    """

    def __init__(self, grass_cover, tree_cover, soil_moisture, grass_biomass, tree_biomass, fire_p=1, fire_distr=1,
                 herd=np.array([0]), preference=np.array([0]), consumed=np.array([0]), forage="herdrelative",
                 infiltration=0.5, evaporation=0.231, unitary_volume_soil_porosity=0.330):

        # initialize all model parameters and initial conditions
        self.grass_cover = grass_cover
        self.tree_cover = tree_cover
        self.grass_biomass = np.maximum(np.zeros(grass_cover.shape), grass_biomass)
        self.tree_biomass = np.maximum(np.zeros(tree_cover.shape), tree_biomass)
        self.soil_moisture = soil_moisture
        self.transpiration_grass = 0.115
        self.colonization_grass = 1.
        self.mortality_grass = 0.019
        self.colonization_tree = 0.019
        self.transpiration_tree = 0.077
        self.mortality_tree = 0.00044
        self.competition_intensity = 0.8
        self.b = 2.5
        self.fire_prob = fire_p / (grass_cover.shape[0] * grass_cover.shape[1])
        self.fire_distr = fire_distr
        self.bare_ground_infiltration = infiltration
        self.uvsp = unitary_volume_soil_porosity
        self.evaporation = evaporation
        self.fireyear = np.ones(grass_cover.shape, dtype='bool')
        self.fire = np.zeros(grass_cover.shape, dtype='bool')
        self.colonization = np.zeros([104, grass_cover.shape[0], grass_cover.shape[1]])
        self.tree_fire_resistance = 0.2
        self.rain_factor_grass = 0.1
        self.rain_factor_tree = 0.2
        self.drought = 4
        self.biomass_factor_grass_1 = 1
        self.biomass_factor_grass_2 = 0
        self.biomass_factor_tree_1 = 1
        self.biomass_factor_tree_2 = 0
        self.biomass_factor_grass_1_wet = 0.6
        self.biomass_factor_grass_2_wet = 0.4
        self.biomass_factor_tree_1_wet = 0.9
        self.biomass_factor_tree_2_wet = 0.1
        self.standing_dead_grass_biomass = grass_cover
        self.standing_dead_tree_biomass = tree_cover
        self.tree_biomass_preyear = 0.5 * tree_cover
        self.grass_increase_steepness = -1
        self.tree_increase_steepness = -0.6
        self.max_grass_week = 0.2
        self.max_tree_week = 0.2
        self.grass_increase_turn = 1
        self.tree_increase_turn = 2
        self.grass_steepness = -15
        self.tree_steepness = -3
        self.grass_turn = 0.4
        self.tree_turn = 0.8
        self.dead_biomass_mortality = 0.95
        self.grazing_factor = 0
        self.browsing_factor = 0
        self.grazing_factor_wet = 0.01
        self.browsing_factor_wet = 0.05
        self.max_grass_increase = 0.15
        self.max_tree_increase = 0.15
        self.grass_increase = 0
        self.tree_increase = 0
        self.herd = herd
        self.preference = preference
        self.consumed = consumed
        self.forage = forage
        rg = RainGenerator(years=1000)
        rg.create_gauss_params()
        rg.let_it_rain()
        self.average_rain = np.mean(rg.weekly_rain, axis=0) * 0.001 + 0.00001

    def infiltration_rate(self):
        # calculate infiltration parameter (see Synodinos et al. 2015)
        return ((1 - self.bare_ground_infiltration) * (self.grass_cover + self.tree_cover)
                + self.bare_ground_infiltration)

    def competition(self):
        # calculate competition parameter (see Synodinos et al. 2015)
        return self.competition_intensity * self.grass_cover ** (self.b * (1 - self.grass_cover))

    def cover_model(self, grazing, browsing, rainfall):
        # each day calculate soil moisture (see Synodinos et al. 2015)
        self.soil_moisture = np.minimum(np.ones(self.grass_cover.shape),
                                        np.maximum(np.zeros(self.grass_cover.shape),
                                                   (self.soil_moisture
                                                    + np.minimum(np.ones(self.grass_cover.shape),
                                                                 self.infiltration_rate() * (rainfall / self.uvsp)
                                                                 * (1 - self.soil_moisture))
                                                    - np.minimum(np.ones(self.grass_cover.shape),
                                                                 self.soil_moisture
                                                                 * (self.evaporation
                                                                    * (1 - self.tree_cover - self.grass_cover)
                                                                    + self.transpiration_grass * self.grass_cover
                                                                    + self.transpiration_tree * self.tree_cover)))))

        # each day calculate grass and tree cover
        self.grass_cover = np.minimum(np.ones(self.grass_cover.shape),
                                      np.maximum(np.zeros(self.grass_cover.shape),
                                                 (self.grass_cover
                                                  + self.colonization_grass * self.soil_moisture * self.grass_cover
                                                  * (1 - self.grass_cover - self.tree_cover)
                                                  * (self.biomass_factor_grass_1 + self.biomass_factor_grass_2
                                                     * np.maximum(np.zeros(self.grass_cover.shape),
                                                                  self.grass_increase) / self.max_grass_increase)
                                                  - (self.mortality_grass + self.grazing_factor
                                                     * np.minimum(np.ones(self.grass_cover.shape),
                                                                  grazing / (self.grass_biomass + 0.1)))
                                                  * self.grass_cover)))

        tree_colonization = self.colonization_tree * self.soil_moisture * self.tree_cover \
                            * (1 - self.tree_cover - self.competition()) \
                            * (self.biomass_factor_tree_1
                               + self.biomass_factor_tree_2
                               * np.maximum(np.zeros(self.tree_cover.shape), self.tree_increase)
                               / self.max_tree_increase)

        self.tree_cover = np.minimum(np.ones(self.tree_cover.shape),
                                     np.maximum(np.zeros(self.tree_cover.shape),
                                                (self.tree_cover + tree_colonization
                                                 - (self.mortality_tree + self.browsing_factor
                                                    * np.minimum(np.ones(self.tree_cover.shape),
                                                                 browsing / (self.tree_biomass + 0.1)))
                                                 * self.tree_cover)))

        # remember tree colonization of last 2 years:
        self.colonization = np.delete(self.colonization, 0, axis=0)
        self.colonization = np.append(self.colonization, [tree_colonization], axis=0)

    def fire_outbreak(self, week):
        # one fire per year:
        if week % 52 == 0:
            fireyear = np.ones(self.grass_cover.shape, dtype='bool')
        else:
            fireyear = self.fireyear

        # fuel:
        fuel = np.minimum(np.ones(self.grass_cover.shape),
                          1 - np.exp(-1.5 * (self.grass_biomass + self.standing_dead_grass_biomass)))

        # seasonality:
        count, bins = np.histogram(np.random.normal(25, 6, 1000), 52, range=(0, 52))
        count = count[np.hstack((np.arange(10, 52), range(10)))]

        # all together - fire probability:
        gamma = fireyear * self.fire_prob * (count / 1000)[week % 52] * fuel
        self.fire = np.array(np.random.uniform(0, 1, fuel.shape) < gamma)

        # fire spread:
        lw, num = measurements.label(
            (np.array(np.random.uniform(0, 1, fuel.shape) < (self.fire_distr * fuel * fireyear)) + self.fire),
            structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        clust = lw[np.where(self.fire)]
        self.fire[np.isin(lw, clust)] = 1

        # remember fire:
        self.fireyear = fireyear * np.abs(self.fire - 1)

        # fire consequences:
        self.fire_consequences(fuel)

    def fire_consequences(self, fuel):
        # tree cover reduced:
        self.tree_cover = np.maximum(np.zeros(self.tree_cover.shape),
                                     self.tree_cover - np.sum(self.colonization, axis=0) * self.fire
                                     - self.tree_fire_resistance * fuel * self.fire * self.tree_cover)

        # tree biomass reduced:
        self.tree_biomass = np.maximum(np.zeros(self.tree_cover.shape),
                                       self.tree_biomass
                                       - self.tree_fire_resistance * fuel * self.fire * self.tree_biomass)

        # grass biomass removed:
        self.grass_biomass = np.abs(self.fire - 1) * self.grass_biomass

        # dead biomass removed:
        self.standing_dead_grass_biomass = np.abs(self.fire - 1) * self.standing_dead_grass_biomass
        self.standing_dead_tree_biomass = np.abs(self.fire - 1) * self.standing_dead_tree_biomass

    def biomass_dry_season(self):
        # no coupling between biomass and cover
        self.biomass_factor_grass_1 = 1
        self.biomass_factor_grass_2 = 0
        self.grazing_factor = 0
        self.biomass_factor_tree_1 = 1
        self.biomass_factor_tree_2 = 0
        self.browsing_factor = 0

        # death of grass biomass
        self.standing_dead_grass_biomass = self.standing_dead_grass_biomass + self.grass_biomass
        self.grass_biomass = np.zeros(self.grass_biomass.shape)
        self.tree_biomass_preyear = self.tree_biomass

    def biomass_wet_season(self, rainfall, week):
        # coupling between biomass and cover
        self.biomass_factor_grass_1 = self.biomass_factor_grass_1_wet
        self.biomass_factor_grass_2 = self.biomass_factor_grass_2_wet
        self.grazing_factor = self.grazing_factor_wet
        self.biomass_factor_tree_1 = self.biomass_factor_tree_1_wet
        self.biomass_factor_tree_2 = self.biomass_factor_tree_2_wet
        self.browsing_factor = self.browsing_factor_wet

        # sigmoid curves for the growth
        self.max_grass_increase = self.max_grass_week / (1 + np.exp(
            self.grass_increase_steepness * (self.grass_biomass - self.grass_increase_turn)))  # 0.0005
        self.max_tree_increase = self.max_tree_week / (1 + np.exp(
            self.tree_increase_steepness * (self.tree_biomass - self.tree_increase_turn)))  # 0.00005
        self.grass_increase = (self.max_grass_increase / (1 + np.exp(
            self.grass_steepness * (self.grass_cover * (rainfall * (1 - self.rain_factor_grass) /
                                                        self.average_rain[week % 52] + self.rain_factor_grass)
                                    - self.grass_turn))))
        self.tree_increase = (self.max_tree_increase / (1 + np.exp(
            self.tree_steepness * (self.tree_cover * (rainfall * (1 - self.rain_factor_tree) /
                                                      self.average_rain[week % 52] + self.rain_factor_tree)
                                   - self.tree_turn))))

        # add the growth to existing biomass
        self.grass_biomass = np.maximum(np.zeros(self.grass_biomass.shape), self.grass_biomass + self.grass_increase)
        self.tree_biomass = np.maximum(np.zeros(self.tree_biomass.shape), self.tree_biomass + self.tree_increase)

    def optimal_foraging(self):
        grazed = np.zeros(self.grass_cover.shape)
        browsed = np.zeros(self.grass_cover.shape)
        grazed_dead = np.zeros(self.grass_cover.shape)
        browsed_dead = np.zeros(self.grass_cover.shape)

        # random shuffle the foraging types
        num_herds = np.arange(len(self.herd))
        np.random.shuffle(num_herds)
        spacesize = np.zeros((len(self.herd), np.max(self.herd)))
        for pos in num_herds:
            pref = self.preference[pos]
            for i in range(self.herd[pos]):
                # calculate foraging window size
                food = (pref * (self.grass_biomass + self.standing_dead_grass_biomass - grazed - grazed_dead) + (
                        1 - pref) * (self.tree_biomass + self.standing_dead_tree_biomass - browsed - browsed_dead))
                if 0.4 * np.median(food[(self.grass_cover + self.tree_cover) > 0]) > 0:
                    space = max(1,
                                min(min(self.grass_cover.shape) // 2 * 2 - 1,
                                    math.ceil(np.sqrt(self.consumed[pos] / (0.4 * np.median(
                                        food[(self.grass_cover + self.tree_cover) > 0])))) // 2 * 2 + 1))
                else:
                    space = min(self.grass_cover.shape) // 2 * 2 - 1

                # find the best available foraging window position
                kernel = diags(np.ones((space * 2 - 1)), np.arange(-space + 1, space),
                               shape=(max(self.grass_cover.shape), max(self.grass_cover.shape))).toarray()
                food = (pref * (self.grass_biomass + self.standing_dead_grass_biomass - grazed - grazed_dead) + (
                        1 - pref) * (self.tree_biomass + self.standing_dead_tree_biomass - browsed - browsed_dead))
                c = kernel[0:self.grass_cover.shape[0], 0:self.grass_cover.shape[0]].dot(
                    food.dot(kernel[0:self.grass_cover.shape[1], 0:self.grass_cover.shape[1]]))
                walk = np.argpartition(-np.ndarray.flatten(c), self.herd[pos])[0]
                food2 = np.zeros((self.grass_cover.shape[0] + space - 1, self.grass_cover.shape[1] + space - 1))
                feed = np.zeros((self.grass_cover.shape[0] + space - 1, self.grass_cover.shape[1] + space - 1))

                # relative foraging within the window size
                food2[int((space - 1) / 2):int(self.grass_cover.shape[0] + (space - 1) / 2),
                      int((space - 1) / 2):int(self.grass_cover.shape[1] + (space - 1) / 2)] = food
                food_local = food2[(walk // (self.grass_cover.shape[1])):(
                        walk // (self.grass_cover.shape[1]) + space),
                             (walk % (self.grass_cover.shape[1])):(
                                     walk % (self.grass_cover.shape[1]) + space)]
                food_local = food_local / np.maximum(0, np.sum(food_local)) * 100
                feed[(walk // (self.grass_cover.shape[1])):(walk // (self.grass_cover.shape[1]) + space),
                     (walk % (self.grass_cover.shape[1])):(walk % (self.grass_cover.shape[1]) + space)] = \
                    feed[(walk // (self.grass_cover.shape[1])):(walk // (self.grass_cover.shape[1]) + space),
                         (walk % (self.grass_cover.shape[1])):(walk % (self.grass_cover.shape[1]) + space)] + food_local

                if space // 2 > 0:
                    feed = feed[space // 2:-(space // 2), space // 2:-(space // 2)]

                # calculate desired food per patch
                grazed_here = pref * feed / 100 * self.consumed[pos]
                browsed_here = (1 - pref) * feed / 100 * self.consumed[pos]
                grazed_dead_here = np.zeros(self.grass_cover.shape)
                browsed_dead_here = np.zeros(self.grass_cover.shape)

                # feeding of fresh and afterwards dead biomass if needed
                if np.any(grazed_here > (self.grass_biomass - grazed)):
                    grazed_dead_here[grazed_here > (self.grass_biomass - grazed)] = \
                        (grazed_here - self.grass_biomass - grazed)[grazed_here > (self.grass_biomass - grazed)]
                    grazed_here[grazed_here > (self.grass_biomass - grazed)] = (self.grass_biomass - grazed)[
                        grazed_here > (self.grass_biomass - grazed)]
                    if np.any(grazed_dead_here > (self.standing_dead_grass_biomass - grazed_dead)):
                        grazed_dead_here[grazed_dead_here > (self.standing_dead_grass_biomass - grazed_dead)] = (
                                self.standing_dead_grass_biomass - grazed_dead)[
                            grazed_dead_here > (self.standing_dead_grass_biomass - grazed_dead)]
                if np.any(browsed_here > (self.tree_biomass - browsed)):
                    browsed_dead_here[browsed_here > (self.tree_biomass - browsed)] = \
                        (browsed_here - self.tree_biomass - browsed)[browsed_here > (self.tree_biomass - browsed)]
                    browsed_here[browsed_here > (self.tree_biomass - browsed)] = (self.tree_biomass - browsed)[
                        browsed_here > (self.tree_biomass - browsed)]
                    if np.any(browsed_dead_here > (self.standing_dead_tree_biomass - browsed_dead)):
                        browsed_dead_here[browsed_dead_here > (self.standing_dead_tree_biomass - browsed_dead)] = (
                                self.standing_dead_tree_biomass - browsed_dead)[
                            browsed_dead_here > (self.standing_dead_tree_biomass - browsed_dead)]

                # sum of the herbivore consumptions
                grazed = grazed + np.maximum(np.zeros(self.grass_cover.shape), grazed_here)
                browsed = browsed + np.maximum(np.zeros(self.grass_cover.shape), browsed_here)
                grazed_dead = grazed_dead + np.maximum(np.zeros(self.grass_cover.shape), grazed_dead_here)
                browsed_dead = browsed_dead + np.maximum(np.zeros(self.grass_cover.shape), browsed_dead_here)
                spacesize[pos, i] = space
        return grazed, browsed, grazed_dead, browsed_dead, spacesize

    def weekly_timestep(self, rainfall, week):
        # grazing and browsing amounts depending on mode:
        if np.sum(self.herd) > 0:
            grazing, browsing, grazing_dead, browsing_dead, spacesize = self.optimal_foraging()

        else:
            grazing = 0
            browsing = 0
            grazing_dead = 0
            browsing_dead = 0
            spacesize = 0

        # cover
        self.cover_model(grazing, browsing, rainfall)

        # fire
        self.fire_outbreak(week)

        # aridity:
        if rainfall == 0:
            self.drought = self.drought + 1
        else:
            self.drought = 0

        if (week % 52) == 46:
            self.biomass_dry_season()
        if (week % 52) < 46 and self.drought < 4:
            self.biomass_wet_season(rainfall, week)

        # death of tree biomass:
        self.tree_biomass = np.maximum(np.zeros(self.tree_biomass.shape),
                                       self.tree_biomass - stats.norm.cdf(np.arange(52), 40, 4)[
                                           np.concatenate([np.arange(38, 52), np.arange(38)])][
                                           week % 52] * self.tree_biomass)
        self.standing_dead_tree_biomass = self.standing_dead_tree_biomass + stats.norm.cdf(np.arange(52), 40, 4)[
            np.concatenate([np.arange(38, 52), np.arange(38)])][week % 52] * self.tree_biomass

        # decomposition of dead biomass:
        self.standing_dead_grass_biomass = np.maximum(np.zeros(self.grass_biomass.shape),
                                                      self.dead_biomass_mortality * self.standing_dead_grass_biomass
                                                      - grazing_dead)
        self.standing_dead_tree_biomass = np.maximum(np.zeros(self.tree_biomass.shape),
                                                     self.dead_biomass_mortality * self.standing_dead_tree_biomass
                                                     - browsing_dead)

        # grazing and browsing on vegetation:
        self.grass_biomass = np.maximum(np.zeros(self.grass_biomass.shape), self.grass_biomass - grazing)
        self.tree_biomass = np.maximum(np.zeros(self.tree_biomass.shape), self.tree_biomass - browsing)
        self.grass_increase = np.maximum(np.zeros(self.grass_biomass.shape), self.grass_increase - grazing)
        self.tree_increase = np.maximum(np.zeros(self.tree_biomass.shape), self.tree_increase - browsing)

        # model output
        return (self.soil_moisture, self.grass_cover, self.tree_cover, self.grass_biomass, self.tree_biomass,
                self.standing_dead_grass_biomass, self.standing_dead_tree_biomass, self.fire, self.herd,
                (grazing + browsing + grazing_dead + browsing_dead), spacesize)
