import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Creation d'une base de commentaires


commentaires = [
    ("Le professeur explique clairement les notions", "pedagogie"),
    ("Les explications sont simples et faciles a suivre", "pedagogie"),
    ("Le cours est bien presente et le professeur est clair", "pedagogie"),
    ("Le professeur donne de bons exemples", "pedagogie"),
    ("Les notions sont bien introduites pendant le cours", "pedagogie"),
    ("Le professeur va trop vite dans ses explications", "pedagogie"),
    ("Le cours est vivant et agreable a suivre", "pedagogie"),
    ("Les explications manquent parfois de details", "pedagogie"),
    ("Le professeur prend le temps de repondre aux questions", "pedagogie"),
    ("Les exemples aident beaucoup a comprendre", "pedagogie"),

    ("Les exercices sont trop difficiles", "difficulte"),
    ("Le niveau du cours est eleve", "difficulte"),
    ("Les notions demandees sont compliquees", "difficulte"),
    ("Le partiel semble plus difficile que les TD", "difficulte"),
    ("Il faut beaucoup travailler pour comprendre", "difficulte"),
    ("Les calculs sont longs et difficiles", "difficulte"),
    ("Le cours demande de bonnes bases en mathematiques", "difficulte"),
    ("Les questions sont parfois piegeuses", "difficulte"),
    ("Les TD sont difficiles mais utiles", "difficulte"),
    ("Le niveau augmente rapidement pendant le semestre", "difficulte"),

    ("Le planning du cours n'est pas tres clair", "organisation"),
    ("Les informations arrivent souvent trop tard", "organisation"),
    ("Les documents ne sont pas toujours disponibles", "organisation"),
    ("L'organisation des seances pourrait etre meilleure", "organisation"),
    ("Les horaires changent parfois au dernier moment", "organisation"),
    ("Le cours manque un peu de structure", "organisation"),
    ("Les consignes ne sont pas toujours precises", "organisation"),
    ("Il faudrait mieux organiser les supports de cours", "organisation"),
    ("Les chapitres ne sont pas toujours dans le bon ordre", "organisation"),
    ("Le calendrier des evaluations devrait etre annonce plus tot", "organisation"),

    ("Il y a beaucoup de devoirs a rendre", "charge_travail"),
    ("La charge de travail est importante", "charge_travail"),
    ("Le cours demande beaucoup de temps personnel", "charge_travail"),
    ("Il faut reviser regulierement pour suivre", "charge_travail"),
    ("Les exercices prennent beaucoup de temps", "charge_travail"),
    ("Le rythme de travail est assez soutenu", "charge_travail"),
    ("Il y a trop de choses a apprendre en peu de temps", "charge_travail"),
    ("La preparation des TD est longue", "charge_travail"),
    ("Le travail demande est assez lourd", "charge_travail"),
    ("Il faut beaucoup s'investir dans cette matiere", "charge_travail"),

    ("Je suis satisfait du cours", "satisfaction"),
    ("Le cours est interessant et utile", "satisfaction"),
    ("J'ai l'impression de progresser", "satisfaction"),
    ("Cette matiere est motivante", "satisfaction"),
    ("Le cours donne envie d'aller plus loin", "satisfaction"),
    ("Je trouve le contenu pertinent", "satisfaction"),
    ("Le cours est globalement positif", "satisfaction"),
    ("J'apprecie la facon dont le cours est construit", "satisfaction"),
    ("Cette matiere est enrichissante", "satisfaction"),
    ("Le cours correspond a mes attentes", "satisfaction"),
]

# On duplique legerement la base avec de petites variantes pour avoir plus d'observations
base = []

for texte, theme in commentaires:
    base.append((texte, theme))
    base.append((texte + " cette annee", theme))
    base.append((texte + " dans cette matiere", theme))

df = pd.DataFrame(base, columns=["commentaire", "theme"])

df.to_csv("donnees/commentaires_etudiants.csv", index=False)

print("Apercu de la base :")
print(df.head())
print("Nombre d'observations :", len(df))
print("Repartition des themes :")
print(df["theme"].value_counts())

# Nettoyage simple du texte


def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-zA-Zàâäéèêëîïôöùûüç\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


df["commentaire_nettoye"] = df["commentaire"].apply(nettoyer_texte)

# Visualisation des themes

plt.figure(figsize=(8, 5))
df["theme"].value_counts().plot(kind="bar")
plt.title("Répartition des thèmes")
plt.xlabel("Thème")
plt.ylabel("Nombre de commentaires")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("sorties/graphiques/repartition_themes.png")
plt.close()

#Preparation du modele


X = df["commentaire_nettoye"]
y = df["theme"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

mots_vides = [
    "le", "la", "les", "un", "une", "des", "de", "du", "ce", "cet", "cette",
    "et", "est", "sont", "a", "au", "aux", "en", "dans", "pour", "par",
    "avec", "sur", "il", "elle", "je", "ne", "pas", "plus", "moins"
]

vectorizer = TfidfVectorizer(stop_words=mots_vides)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# Modele de classification


modele = LogisticRegression(max_iter=1000)
modele.fit(X_train_tfidf, y_train)

predictions = modele.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy du modele :", round(accuracy, 3))
print("Rapport de classification :")
print(classification_report(y_test, predictions))
print("Matrice de confusion :")
print(confusion_matrix(y_test, predictions))



# Sauvegarde des predictions


resultats = pd.DataFrame({
    "commentaire": X_test,
    "theme_reel": y_test,
    "theme_predit": predictions
})
resultats.to_csv("sorties/predictions_commentaires.csv", index=False)





mots = vectorizer.get_feature_names_out()
classes = modele.classes_
importance = []

for i, classe in enumerate(classes):
    coefficients = modele.coef_[i]
    indices = np.argsort(coefficients)[-8:]
    for indice in indices:
        importance.append({
            "theme": classe,
            "mot": mots[indice],
            "coefficient": coefficients[indice]
        })

importance_df = pd.DataFrame(importance)
importance_df.to_csv("sorties/mots_importants.csv", index=False)
print("Mots importants par theme :")
print(importance_df)



# Graphique simple des mots importants


for classe in classes:
    temp = importance_df[importance_df["theme"] == classe].sort_values("coefficient")
    plt.figure(figsize=(8, 5))
    plt.barh(temp["mot"], temp["coefficient"])
    plt.title("Mots importants - " + classe)
    plt.xlabel("Coefficient")
    plt.ylabel("Mot")
    plt.tight_layout()
    plt.savefig("sorties/graphiques/mots_importants_" + classe + ".png")
    plt.close()
