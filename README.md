# Classification de commentaires étudiants avec NLP

## Présentation du projet

Ce projet a pour objectif de classer automatiquement des commentaires étudiants selon leur thème principal.
L'idée est de partir de commentaires écrits en langage naturel, puis d'utiliser des méthodes simples de traitement de texte pour identifier le sujet du commentaire.
Chaque commentaire est classé dans une des catégories suivantes :

- pédagogie 
- difficulté 
- organisation 
- charge de travail 
- satisfaction.

Le projet est réalisé avec Python.

## Objectif

L'objectif est de construire un modèle capable de prédire le thème d'un commentaire étudiant.
Par exemple, un commentaire comme :

`Le professeur explique clairement les notions`

sera classé dans le thème :
`pédagogie`.

Un commentaire comme :
`Les exercices sont trop difficiles`

sera plutôt classé dans le thème :
`difficulté`.

## Méthode

Le projet suit plusieurs étapes :

1. création d'une base simulée de commentaires étudiants 
2. nettoyage simple du texte 
3. transformation des commentaires avec la méthode TF-IDF 
4. entraînement d'un modèle de classification 
5. évaluation du modèle 
6. analyse des mots les plus importants par thème 
7. export des résultats et des graphiques

## Modèle utilisé

Le modèle utilisé est une régression logistique.
Avant l'estimation du modèle, les textes sont transformés en variables numériques avec la méthode TF-IDF.
Cette méthode permet de donner plus de poids aux mots importants dans les commentaires, tout en réduisant l'importance des mots trop fréquents.

## Résultats

La base contient 150 commentaires simulés, répartis de manière équilibrée entre les cinq thèmes.
Le modèle obtient une accuracy d'environ 0,97 sur l'échantillon de test.
Cela signifie qu'il classe correctement la grande majorité des commentaires.

Les mots importants identifiés par le modèle sont cohérents avec les thèmes :

- pour la charge de travail : `travail`, `beaucoup`, `temps` 
- pour la difficulté : `difficiles`, `niveau`, `piegeuses` 
- pour l'organisation : `supports`, `consignes`, `toujours` 
- pour la pédagogie : `professeur`, `explications`, `exemples` 
- pour la satisfaction : `cours`, `motivante`, `enrichissante`

## Fichiers générés

Le programme génère plusieurs fichiers :

- `donnees/commentaires_etudiants.csv` : base de commentaires simulés 
- `sorties/predictions_commentaires.csv` : prédictions du modèle 
- `sorties/mots_importants.csv` : mots les plus importants par thème 
- `sorties/graphiques/repartition_themes.png` : répartition des thèmes 
- `sorties/graphiques/mots_importants_*.png` : graphiques des mots importants par thème

## Interprétation

Ce projet montre comment des commentaires écrits peuvent être transformés en données exploitables.
Même avec un modèle simple, il est possible de classer automatiquement des textes courts selon leur thème principal.
Ce type d'approche peut être utilisé pour analyser des retours d'étudiants, des enquêtes de satisfaction ou des commentaires libres.