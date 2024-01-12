# GTFS4EV
**GTFS for Electric Vehicles**

*Simulation of the spatial and temporal electric energy demand of a an electric vehicle fleet using  GTFS data to model the vehicle behaviour.*

authors = Jeremy Dumoulin, Nicolas Wyrsch

email = jeremy.dumoulin@epfl.ch 

langage = python 3


Todo

- Transférer tout ce qui relève de la modélisation dans la classe TripSim
- Propriétés de la classe : single_trip_length_km, single_trip_duration_sec
- Methodes : speed, profile, métrique pour chaque

- Créer un notebook jupyter qui permet de visualiser un trip en spécifiant son id et/ou un point
- Visualiser l'ensemble du réseau

- Renommer la classe Vehicle en tripSim

- Ajouter les données pour un single_trip_data
- Mettre d'autres propriétés comme le trip_duration, trip_length, d'autres valeurs de l'article

- Extraire le VKT

- Infos spatiales : taille de la bbox
- Visualisation des routes et des stops à la manière de ev-fleet-sim
- Calculer la bbox avec ev-fleet-sim et comparer 
- Mapper les distances avec les routes OSM
- vérifier les projections et laisser à l'utilisateur la possibilité d'adapter le système de projection
- Vérifier qu'il existe des fréquences pour chaque trip


- Croiser avec des informations GIS : pourcentage de routes desservies par le reseau, accessibilite 
- GTFS Feed. Implémenter une fonction qui permet de filtrer le dataset selon le service ou les routes ou autre. filter_services
- Améliorer les fonctions qui permettent de vérifier la consistance des données. Regarder tous ce qui est lié aux trips, aux stops, aux routes