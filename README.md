# perception-numpy

Briques de perception par capteurs et de positionnement, réimplémentées avec NumPy
uniquement, chacune accompagnée de tests qui la comparent à un résultat connu.

Statut : en construction. Ce dépôt ne contient pour l'instant que la structure ;
chaque module est ajouté avec ses tests, pas avant.

## Périmètre visé

| Domaine | Briques |
|---|---|
| Modèle de caméra | projection, rétroprojection, champ de vision |
| Réseaux de neurones | softmax, entropie croisée, convolution, pooling |
| Métriques | IoU, GIoU, précision/rappel, matrice de confusion, mIoU |
| Détection | suppression des non-maxima (NMS), appariement, précision moyenne |
| Positionnement GNSS | moindres carrés de Gauss-Newton sur pseudodistances |

## Principes

- NumPy seul pour les algorithmes ; aucune bibliothèque de vision ou d'apprentissage dans le cœur.
- Chaque fonction a un test contre un cas calculé à la main ou une référence indépendante.
- Aucune donnée, aucun sujet ni code tiers n'est inclus.

## Développement

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
```

## Licence

MIT.
