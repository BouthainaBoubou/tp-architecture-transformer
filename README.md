# Architecture Transformer : exercices pratiques

Ce notebook contient deux parties :

1. **Le cours** : comparer RNN, LSTM et Transformer sur la même tâche, puis utiliser GPT-2 et BERT déjà entraînés.
2. **Les exercices** : appliquer le Transformer à trois problèmes.

| Exercice | Sujet | Type de modèle |
|---|---|---|
| 1 | Prédire la consommation électrique de l'heure suivante | Transformer Encoder (série temporelle) |
| 2 | Générer du texte caractère par caractère | Mini-GPT (attention causale) |
| 3 | Retrouver des mots masqués dans une phrase | Mini-BERT (attention bidirectionnelle) |

---

## Fichiers

```
architecture_transformer.ipynb      # le notebook
household_power_consumption.txt     # données de l'exercice 1
corpus.txt                          # texte de l'exercice 2
README.md
```

Les deux fichiers de données doivent être **dans le même dossier que le notebook**.

- `household_power_consumption.txt` : dataset *Individual Household Electric Power Consumption* (UCI Machine Learning Repository).
- `corpus.txt` : quelques pages de texte en français.

---

## Installation

```bash
pip install torch numpy pandas matplotlib transformers
```

`transformers` sert à charger GPT-2 et BERT dans la partie cours. Il faut une connexion internet au premier lancement pour télécharger les modèles.

---

## Exécution

Ouvrir le notebook et exécuter les cellules **dans l'ordre**, de haut en bas.

---

## Partie cours

### 1. Même tâche pour les trois modèles

Les trois modèles reçoivent **10 valeurs d'une sinusoïde** et prédisent **la valeur suivante**.

- Données : 200 points de `sin(x)` entre 0 et 20, soit 190 exemples.
- Découpage dans l'ordre du temps : 150 exemples pour l'entraînement, 40 pour le test.
- Entraînement : perte MSE, optimiseur Adam, 300 époques.

### 2. RNN

Le RNN lit la séquence valeur par valeur et met à jour une **mémoire** (état caché de 16 nombres). La dernière mémoire passe dans une couche `Linear(16, 1)` qui donne la prédiction.

**Limite :** il retient mal les informations lointaines (gradients qui disparaissent ou explosent).

### 3. LSTM

Le LSTM ajoute des **portes** qui décident quoi oublier, quoi ajouter et quoi transmettre. Il garde ainsi mieux l'information sur de longues séquences.

### 4. Transformer

Le Transformer ne lit pas valeur par valeur : il regarde **toute la séquence en même temps**. Grâce à l'**attention**, chaque position choisit les autres positions utiles. Un **encodage positionnel** lui indique l'ordre des valeurs.

**Limite :** le coût de l'attention augmente avec le carré de la longueur de la séquence.

### Comparaison sur le test

| Modèle | Erreur de test (MSE) | Valeur réelle | Valeur prédite |
|---|---|---|---|
| RNN | 0.000028 | -0.3639 | -0.3601 |
| LSTM | 0.000003 | -0.3639 | -0.3622 |
| Transformer | 0.000006 | -0.3639 | -0.3662 |

Les trois modèles suivent très bien la sinusoïde. Le LSTM et le Transformer font mieux que le RNN simple.

### 5. GPT-2 (modèle pré-entraîné)

GPT-2 **génère la suite d'un texte** : il prédit le token suivant en ne regardant que les tokens précédents.

Exemple : à partir de *« Artificial intelligence can help people »*, il écrit la suite de la phrase.

### 6. BERT (modèle pré-entraîné)

BERT **retrouve un mot masqué** en utilisant le contexte à gauche **et** à droite (modèle `bert-base-multilingual-cased`).

Exemple : `Rabat est la [MASK] Maroc.` donne **capitale** (42,9 %).

---

## Exercice 1 : consommation électrique

**Objectif :** utiliser les 24 dernières heures pour prédire la puissance moyenne (kW) de l'heure suivante.

**Étapes :**
1. Agrégation des mesures par heure, découpage 70 / 15 / 15 (train / validation / test), normalisation sur l'entraînement.
2. Modèle `EnergyTransformer` : `Linear(1, 32)`, encodage positionnel, 2 couches Transformer Encoder, `Linear(32, 1)` sur la dernière heure.
3. Entraînement avec Adam (lr = 0.001) et une perte MSE, sur 20 époques, en gardant le meilleur modèle de validation.
4. Évaluation en kW et comparaison avec deux références.

**Résultats sur le test :**

| Modèle | MAE (kW) | RMSE (kW) |
|---|---|---|
| Transformer | 0.340 | 0.491 |
| Persistance (dernière heure) | 0.389 | 0.588 |
| Journalière (24 h avant) | 0.513 | 0.764 |

Le Transformer fait mieux que les deux références.

---

## Exercice 2 : mini-GPT

**Objectif :** prédire le caractère suivant et générer du texte.

**Étapes :**
1. Bloc GPT : LayerNorm, attention multi-têtes causale, connexion résiduelle, LayerNorm, MLP 64 → 256 → 64 (GELU), connexion résiduelle.
2. Modèle : embeddings de caractères et de positions, 2 blocs, 4 têtes, projection vers le vocabulaire.
3. Entraînement avec l'entropie croisée sur toutes les positions.
4. Génération de 200 caractères avec les températures 0.5, 1.0 et 1.5.

**Vérification :** modifier un caractère futur ne change pas les prédictions des positions précédentes (masque causal correct).

**Résultat :** plus la température est basse, plus le texte est régulier et répétitif ; plus elle est haute, plus il est varié et contient d'erreurs.

---

## Exercice 3 : mini-BERT

**Objectif :** retrouver des mots masqués grâce au contexte à gauche et à droite.

**Étapes :**
1. Vocabulaire par mots, avec les tokens spéciaux `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]` et `[MASK]`, et des phrases complétées à 32 tokens.
2. Masquage de 15 % des mots : 80 % remplacés par `[MASK]`, 10 % par un mot aléatoire, 10 % conservés.
3. Modèle : embeddings de dimension 64, 2 couches Transformer Encoder, 4 têtes, sans masque causal, avec masque de padding.
4. Perte calculée seulement sur les positions sélectionnées.

**Résultats :**
- Exactitude sur la validation : **86,1 %**.
- Exemple : `le chat [MASK] sur le canapé` donne **dort** (66,7 %).
- Changer le contenu du padding ne modifie pas les prédictions (masque de padding correct).

---

## À retenir

| | GPT | BERT |
|---|---|---|
| Attention | Causale (regarde seulement à gauche) | Bidirectionnelle (gauche et droite) |
| Tâche | Prédire le token suivant | Retrouver un token masqué |
| Usage | Génération de texte | Compréhension de texte |