# Suivi d'objets par Contours Actifs paramétriques

## Description

Ce projet est une implémentation d'un **algorithme de suivi d'objets basé sur les contours actifs** appliqué à l'analyse de mouvements dans des vidéos de football. Il utilise des techniques de filtrage d'image, de détection de gradient et de déformation de contours pour détecter et suivre les objets en mouvement.

## Principe de l'algorithme
<img width="538" alt="schema" src="https://github.com/user-attachments/assets/74cc14cc-69f2-4015-877f-0706a0e3f5bb" />

L'algorithme repose sur l'évolution d'un contour initial selon les forces **internes et externes** définies par :

- Une énergie interne qui contrôle la **souplesse et la rigidité** du contour.
- Une énergie externe qui dépend des gradients de l'image pour attirer le contour vers les bords des objets.

### Formulation mathématique

<img width="538" alt="formulation" src="https://github.com/user-attachments/assets/68b3c3e3-5169-456c-8e76-c98d8ff16206" />

Le contour actif évolue de manière à minimiser la fonction coût ci-dessus.

## Améliorations possibles

- **Amélioration du prétraitement** : Ajout de filtres adaptatifs.
- **Suivi multi-objet** : Gestion plus robuste des collisions.
- **Correction par filtre de Kalman** pour compenser les pertes temporaires.

## Références

📖 Kass, M., Witkin, A., Terzopoulos, D. "Snakes: Active contour models" (1988)  
📖 Caselles, V., Catté, F., Coll, T. "A geometric model for active contours" (1993)  
📖 Cui et al. "SportsMOT: A Large Multi-Object Tracking Dataset" (2023)


