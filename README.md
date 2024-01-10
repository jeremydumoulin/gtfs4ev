# GTFS4EV
**GTFS for Electric Vehicles**

*Simulation of the spatial and temporal electric energy demand of a an electric vehicle fleet using  GTFS data to model the vehicle behaviour.*

authors = Jeremy Dumoulin, Nicolas Wyrsch

email = jeremy.dumoulin@epfl.ch 

langage = python 3


Todo

- GTFS Feed. Implémenter des fonctions pour analyser le feed : Fonction data_check
- GTFS Feed. Implémenter une fonction qui permet de clean les différentes erreurs. Implementer aussi une fonction globale qui permet de faire un clean_all
- GTFS Feed. Implémenter une fonction qui permet de filtrer le dataset selon le service ou les routes ou autre. filter_services

- GTFS Feed. Extraire quelques données topologiques :
	- Ratio entre le nombre de stops et le nombre de routes et de trips
	- Frequence des stops
	- Longueur des trips
	- Topologie : nombre de noeuds, nombre de links (comparer), longueur totale des trips et des routes

- GTFS Feed. Ajouter epsg comme paramètre


- Créer un notebook jupyter qui permet de visualiser un trip en spécifiant son id et/ou un point
- Visualiser l'ensemble du réseau

- Renommer la classe Vehicle en tripSim

- Infos spatiales : taille de la bbox
- Visualisation des routes et des stops à la manière de ev-fleet-sim
- Calculer la bbox avec ev-fleet-sim et comparer 
- Mapper les distances avec les routes OSM
- vérifier les projections et laisser à l'utilisateur la possibilité d'adapter le système de projection
- Vérifier qu'il existe des fréquences pour chaque trip


- Croiser avec des informations GIS : pourcentage de routes desservies par le reseau, accessibilite 