# Suivi d'objets avec Contours Actifs

## Description

Ce projet est une implémentation d'un **algorithme de suivi d'objets basé sur les contours actifs** appliqué à l'analyse de mouvements dans des vidéos de football. Il utilise des techniques de filtrage d'image, de détection de gradient et de déformation de contours pour détecter et suivre les objets en mouvement.

## Principe de l'algorithme
<img width="538" alt="schema" src="https://github.com/user-attachments/assets/74cc14cc-69f2-4015-877f-0706a0e3f5bb" />

L'algorithme repose sur l'évolution d'un contour initial selon les forces **internes et externes** définies par :

- Une énergie interne qui contrôle la **souplesse et la rigidité** du contour.
- Une énergie externe qui dépend des gradients de l'image pour attirer le contour vers les bords des objets.

### Formulation mathématique

<img width="538" alt="schema" src="https://github.com/user-attachments/assets/7bd08070-a791-415a-ada9-e0b913d9d393" />

Le contour actif évolue de manière à minimiser la fonction coût ci-dessus.

## Installation

### Prérequis

Assurez-vous d'avoir **Python 3.8+** et installez les dépendances requises :

```bash
pip install numpy matplotlib scipy scikit-image opencv-python
```

## Utilisation

### 1. Exécuter l'algorithme sur un ensemble d'images

1. Placez vos images dans le dossier **dataset/foot1/**.
2. Lancez le programme principal :

```bash
python suivi_objet.py
```

### 2. Tester l'algorithme

Un script de test est prévu pour évaluer l'algorithme sur différentes images :

```bash
python test_suivi.py
```

> *Figure 2 : Suivi de deux joueurs en temps réel*&#x20;

## Résultats

L'algorithme est capable de : ✅ Suivre les objets en mouvement ✅ S'adapter aux déformations des joueurs ✅ Fonctionner sur plusieurs images successives

Cependant, il présente quelques **limitations** : ❌ Sensible à la qualité des images ❌ Peut perdre le suivi en cas d'obstruction

> *Figure 3 : Résultat sur une séquence de football*&#x20;

## Améliorations possibles

- **Amélioration du prétraitement** : Ajout de filtres adaptatifs.
- **Suivi multi-objet** : Gestion plus robuste des collisions.
- **Correction par filtre de Kalman** pour compenser les pertes temporaires.

## Références

📖 Kass, M., Witkin, A., Terzopoulos, D. "Snakes: Active contour models" (1988) 📖 Caselles, V., Catté, F., Coll, T. "A geometric model for active contours" (1993) 📖 Cui et al. "SportsMOT: A Large Multi-Object Tracking Dataset" (2023)

---

🛠 **Projet développé par Arthur De Rouck**

