# Savanna-Vegetation-Model-BioSaVe

This repository supplements our manuscript entitled "Balanced functional herbivore composition stabilizes tree-grass coexistence and productivity in a simulated savanna rangeland ecosystem."
Coauthors: Dirk Lohmann, Morgan Hauptfleisch, Florian Jeltsch

The model simulates grass and tree cover as well as grass and tree biomass of a savanna landscape. It includes ecohydrological dynamics based on Synodinos et al. 2015. The modelled landscape has a resolution of one hectare with a total size of 10000 ha (100 km2), representing the typical scale of a rangeland farm in the area of southern Africa. We use the model to analyze the impact of different herbivore compositions grazing and browsing on the vegetation. Further detail on the model is given in the publication.  

The model is programmed in python. The main model code can be found in the file landscape_herbivores_review.py. To simulate stochasticrain timeseries reflecting typical rainfall patterns, the \italics{RainGenerator}.py is used. The file run_model_fast.py runs the model for a defined scenario and time period and saves the output. The file run_scenarios_all_review.py is responsible for creating scenarios and can run them in parallel. To calculate specific Etosha Heights scenarios (study area in northern Namibia), the initialization file EH_Initialisation.npy is needed.
