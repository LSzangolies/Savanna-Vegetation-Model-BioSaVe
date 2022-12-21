# Initialization and Parallelization to run scenarios in BioSaVe: Large-Scale, Dynamic, Spatial, Ecohydrological
# Vegetation Model for Savanna Tree and Grass Cover and Biomass with herbivores
# copyright Leonna Szangolies, Florian Jeltsch, University of Potsdam

from run_model_fast import run_model
import numpy as np
import multiprocessing as mp
import os

###########################################################################
# General initializations:
lengthx = 100  # for Etosha Heights Simulation: 190
lengthy = 100  # for Etosha Heights Simulation: 647
space = lengthx * lengthy  # for Etosha Heights Simulation: 46500
reps = 10
cores = 10
YEARS = 100
runtime = YEARS * 52
pathcurrent = os.getcwd()

###########################################################################
# Herbivore scenario initialization for Etosha Height:
# Wildlife:
herd = (np.array([580, 30, 12, 16, 16]) / 46500 * space).astype(int)
preference = np.array([0.1, 0.6, 0.2, 0.8, 0.7])
amount = np.array([2 * 20, 40 * 80, 20 * 200, 30 * 150, 10 * 300]) / 250 * 0.08
wild = [herd, preference, amount]
# Livestock:
herd = (np.array([23]) / 46500 * space).astype(int)
preference = np.array([0.9])
amount = np.array([100 * 250]) / 250 * 0.08
live = [herd, preference, amount]

###########################################################################
# Artificial landscape initialization:
soil_moisture = np.array([[0.15]])
grass = np.random.rand(lengthx, lengthy) * 0.5 + 0.3
trees = np.random.rand(lengthx, lengthy) * 0.5 + 0.0
poros = np.full((lengthx, lengthy), 0.33)
infil = np.full((lengthx, lengthy), 0.5)
evap = np.full((lengthx, lengthy), 12 / 52)
grass_biomass = grass * 0
tree_biomass = trees * 0
fire_p = 1
fire_distr = 1
normal = [soil_moisture, grass, trees, poros, infil, evap, grass_biomass, tree_biomass, fire_p, fire_distr]

###########################################################################
# Etosha Heights landscape initilization:
landtype = np.load("EH_Initialisation.npy")

infiltration = np.zeros(landtype.shape)
infiltration[landtype == 10] = 0.05
infiltration[landtype == 7] = 0.5
infiltration[landtype == 2] = 0.75
infiltration[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 5,
                                   landtype == 6, landtype == 8, landtype == 9))] = 0.6

evaporation = np.zeros(landtype.shape)
evaporation[landtype == 10] = 6 / 52
evaporation[landtype == 7] = 12 / 52
evaporation[landtype == 2] = 8 / 52
evaporation[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 5,
                                  landtype == 6, landtype == 8, landtype == 9))] = 10 / 52

porosity = np.ones(landtype.shape)
porosity[landtype == 10] = 0.12
porosity[landtype == 7] = 0.5
porosity[landtype == 2] = 0.16
porosity[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 5,
                               landtype == 6, landtype == 8, landtype == 9))] = 0.33

grass = np.zeros(landtype.shape)
grass[landtype == 10] = np.random.rand(len(grass[landtype == 10])) * 0.15 + 0.05
grass[np.logical_or(landtype == 7, landtype == 9)] = \
    np.random.rand(len(grass[np.logical_or(landtype == 7, landtype == 9)])) * 0.2 + 0.7
grass[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 6))] = \
    np.random.rand(len(grass[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 6))])) \
    * 0.2 + 0.5
grass[np.logical_or.reduce((landtype == 0, landtype == 2, landtype == 8))] = \
    np.random.rand(len(grass[np.logical_or.reduce((landtype == 0, landtype == 2, landtype == 8))])) * 0.2 + 0.3
grass[landtype == 5] = np.random.rand(len(grass[landtype == 5])) * 0.1 + 0.01

trees = np.zeros(landtype.shape)
trees[landtype == 10] = np.random.rand(len(grass[landtype == 10])) * 0.2 + 0.3
trees[np.logical_or(landtype == 7, landtype == 9)] = \
    np.random.rand(len(grass[np.logical_or(landtype == 7, landtype == 9)])) * 0.1 + 0.01
trees[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 6))] = \
    np.random.rand(len(grass[np.logical_or.reduce((landtype == 1, landtype == 3, landtype == 4, landtype == 6))])) \
    * 0.2 + 0.1
trees[np.logical_or.reduce((landtype == 0, landtype == 2, landtype == 8))] = \
    np.random.rand(len(grass[np.logical_or.reduce((landtype == 0, landtype == 2, landtype == 8))])) * 0.2 + 0.15
trees[landtype == 5] = np.random.rand(len(grass[landtype == 5])) * 0.2 + 0.2

soil_moisture = np.array([[0.15]])
grass_biomass = grass * 0
tree_biomass = trees * 0
fire_p = 1
fire_distr = 1

EH = [soil_moisture, grass, trees, porosity, infiltration, evaporation, grass_biomass, tree_biomass, fire_p, fire_distr]

############################################################################
# Design scenarios
Scenario = []
f = 1  # 0 for no fire
forage = "herdrelative"

for perc in np.arange(0, 1.1, 0.2):
    for lsu in [250, 350, 500]:
        for hs in [25]:
            for rep in range(reps):
                # initialize landscape from above (e.g., insert EH instead of normal)
                soil_moisture, grass, trees, poros, infil, evap, grass_biomass, tree_biomass, fire_p, fire_distr = normal
                # To generate new random landscape for each repetition:
                # grass = np.random.rand(lengthx, lengthy) * 0.5 + 0.3
                # trees = np.random.rand(lengthx, lengthy) * 0.5 + 0.0

                # herbivore scenarios on artificial landscape:
                herd = np.array([int(perc * lsu / hs), int((1 - perc) * lsu / hs)])
                preference = np.array([0.2, 0.9])
                amount = np.array([0.14 * 0.45 * hs, 0.14 * 0.45 * hs])

                # output location
                path = pathcurrent + "/ADD/" + forage + "_Fire" + str(f) + "_LSU" + str(lsu) + "_Herds" + str(hs) \
                       + "_Perc" + str(perc) + "/Repetition" + str(rep)

                # create scenario list
                s = np.random.randint(2 ** 31)
                Scenario.append([grass, trees, soil_moisture, grass_biomass, tree_biomass, fire_p, fire_distr,
                                 herd, preference, amount, forage, infil, evap, poros, path, YEARS, space, s])

###########################################################################
# parallelization:
pool = mp.Pool(cores)
pool.starmap(run_model, [(row,) for row in Scenario])
pool.close()
