# perception-numpy

Briques de perception par capteurs et de positionnement, réimplémentées avec NumPy
uniquement, chacune accompagnée de tests qui la comparent à un résultat connu.

Statut : en construction. Chaque module n'est ajouté qu'avec ses tests.

## Origine

Travail universitaire : les algorithmes ont été écrits lors d'exercices de formation d'ingénieur.
Ce dépôt en est une mise en forme pour publication : formatage, documentation et tests ont été
repris et rédigés avec l'aide d'un assistant IA ; la logique des fonctions est celle des exercices.
Les énoncés, squelettes de code et données fournis par la formation ne sont pas inclus.

## Contenu

| Module | Briques |
|---|---|
| `camera` | projection sténopé, rétroprojection, champ de vision |
| `nn` | softmax stable, entropie croisée, convolution 2D (pas, remplissage, dilatation), max pooling, champ réceptif |
| `boxes` | IoU matriciel, GIoU, encodage/décodage par ancres, suppression des non-maxima (NMS) |
| `metrics` | précision/rappel, fréquences et poids de classes, matrice de confusion, IoU/mIoU, appariement des détections, courbe précision-rappel, précision moyenne |

À venir : positionnement GNSS par moindres carrés (Gauss-Newton) sur pseudodistances.

## Principes

- NumPy seul pour les algorithmes ; aucune bibliothèque de vision ou d'apprentissage dans le cœur.
- Chaque fonction a un test contre une valeur calculée à la main (26 tests).
- Aucune donnée, aucun sujet ni code tiers n'est inclus.

## Développement

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
```

## Licence

MIT.
