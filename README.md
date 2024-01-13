# GTFS4EV
**GTFS for Electric Vehicles**

*Simulation of the spatial and temporal electric energy demand of a an electric vehicle fleet using  GTFS data to model the vehicle behaviour.*

authors = Jeremy Dumoulin, Nicolas Wyrsch

email = jeremy.dumoulin@epfl.ch 

langage = python 3


Todo

- Infos spatiales : taille de la bbox
- Visualisation des routes et des stops à la manière de ev-fleet-sim
- Calculer la bbox avec ev-fleet-sim et comparer 
- Mapper les distances avec les routes OSM
- vérifier les projections et laisser à l'utilisateur la possibilité d'adapter le système de projection
- Vérifier qu'il existe des fréquences pour chaque trip


- Croiser avec des informations GIS : pourcentage de routes desservies par le reseau, accessibilite 
- GTFS Feed. Implémenter une fonction qui permet de filtrer le dataset selon le service ou les routes ou autre. filter_services
- Améliorer les fonctions qui permettent de vérifier la consistance des données. Regarder tous ce qui est lié aux trips, aux stops, aux routes

- Vérifier le régime transitoire
- Commenter davantage les classes et méthodes de classe

- Ajouter le calcul CO2 et Particules fines : Estimating public transport emissions from General Transit Feed Specification data
- Avoir une visualisation spatio-temporelle 