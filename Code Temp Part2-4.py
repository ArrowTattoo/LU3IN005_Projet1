import numpy as np
import matplotlib.pyplot as plt

# Dimensions de la grille
GRID_SIZE = 10

# Identifiants des bateaux
SHIP_IDS = {
    'porte_avions': 1,
    'croiseur': 2,
    'contre_torpilleurs': 3,
    'sous_marin': 4,
    'torpilleur': 5
}

# Longueurs des bateaux
SHIP_LENGTHS = {
    1: 5,
    2: 4,
    3: 3,
    4: 3,
    5: 2
}

def peut_placer(grille, bateau, position, direction):
    """
    Vérifie si un bateau peut être placé sur la grille à la position et dans la direction spécifiées.
    """
    x, y = position
    length = SHIP_LENGTHS[bateau]

    if direction == 1:  # Horizontale
        if y + length > GRID_SIZE:
            return False
        for i in range(length):
            if grille[x, y + i] != 0:
                return False
    elif direction == 2:  # Verticale
        if x + length > GRID_SIZE:
            return False
        for i in range(length):
            if grille[x + i, y] != 0:
                return False
    else:
        return False

    return True

def place(grille, bateau, position, direction):
    """
    Place un bateau sur la grille à la position et dans la direction spécifiées.
    """
    x, y = position
    length = SHIP_LENGTHS[bateau]

    if direction == 1:  # Horizontale
        for i in range(length):
            grille[x, y + i] = bateau
    elif direction == 2:  # Verticale
        for i in range(length):
            grille[x + i, y] = bateau

    return grille

def place_alea(grille, bateau):
    """
    Place aléatoirement un bateau sur la grille jusqu'à ce que le placement soit valide.
    """
    while True:
        x = np.random.randint(0, GRID_SIZE)
        y = np.random.randint(0, GRID_SIZE)
        direction = np.random.randint(1, 3)  # 1 pour horizontale, 2 pour verticale

        if peut_placer(grille, bateau, (x, y), direction):
            grille = place(grille, bateau, (x, y), direction)
            break

    return grille

def affiche(grille):
    """
    Affiche la grille de jeu en utilisant matplotlib.
    """
    plt.figure()
    plt.imshow(grille, aspect="equal", cmap="viridis", origin="upper")
    plt.axis('off')
    plt.show()

def eq(grilleA, grilleB):
    """
    Compare deux grilles pour vérifier si elles sont égales.
    """
    return np.array_equal(grilleA, grilleB)

def genere_grille():
    """
    Génère une grille comprenant tous les bateaux disposés de manière aléatoire.
    """
    grille = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for bateau in SHIP_IDS.values():
        grille = place_alea(grille, bateau)

    return grille

# Générer une grille pour chaque joueur
# grille_joueur1 = genere_grille()
# grille_joueur2 = genere_grille()

# # A
# B

# # Afficher les grilles des deux joueurs
# print("Grille du Joueur 1:")
# #affiche(grille_joueur1)
# print(grille_joueur1)

# print("Grille du Joueur 2:")
# #affiche(grille_joueur2)
# print(grille_joueur2)

##### 2 Combinatoire du jeu
# 2.2 Calcul du nombre de placements dans la grille

def nombre_de_facons_placer(grille, bateau):
    """
    Calcule le nombre total de façons de placer un bateau donné sur une grille vide.
    """
    longueur_bateau = SHIP_LENGTHS[bateau]
    count = 0

    # Cas de placement horizontal (direction = 1)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE - longueur_bateau + 1):  # Limite en colonne
            count += 1

    # Cas de placement vertical (direction = 2)
    for x in range(GRID_SIZE - longueur_bateau + 1):  # Limite en ligne
        for y in range(GRID_SIZE):
            count += 1

    return count

grille_vide = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
bateau = 1  # Par exemple, porte-avions (longueur 5)
nombre_facons = nombre_de_facons_placer(grille_vide, bateau)
print(f"Nombre de façons de placer le bateau {bateau}: {nombre_facons}")

bateau = 2
nombre_facons = nombre_de_facons_placer(grille_vide, bateau)
print(f"Nombre de façons de placer le bateau {bateau}: {nombre_facons}")

bateau = 3
nombre_facons = nombre_de_facons_placer(grille_vide, bateau)
print(f"Nombre de façons de placer le bateau {bateau}: {nombre_facons}")

bateau = 4
nombre_facons = nombre_de_facons_placer(grille_vide, bateau)
print(f"Nombre de façons de placer le bateau {bateau}: {nombre_facons}")

bateau = 5
nombre_facons = nombre_de_facons_placer(grille_vide, bateau)
print(f"Nombre de façons de placer le bateau {bateau}: {nombre_facons}")

def nombre_total_facons(grille):
    """
    Calcule le nombre total de façons de placer tous les bateaux sur une grille vide.
    """
    total_facons = 1

    for bateau in SHIP_LENGTHS:
        facons_bateau = nombre_de_facons_placer(grille, bateau)
        total_facons *= facons_bateau

    return total_facons

grille_vide = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
nombre_total = nombre_total_facons(grille_vide)
print(f"Nombre total de façons de placer tous les bateaux: {nombre_total}")

#####
# 2.3 Calcul du nombre de façons de placer une liste de bateaux sur une grille

def nombre_de_facons_placer(grille, bateau):
    """
    Calcule le nombre total de façons de placer un bateau donné sur une grille vide.
    """
    longueur_bateau = SHIP_LENGTHS[bateau]
    count = 0

    # Cas de placement horizontal (direction = 1)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE - longueur_bateau + 1):  # Limite en colonne
            count += 1

    # Cas de placement vertical (direction = 2)
    for x in range(GRID_SIZE - longueur_bateau + 1):  # Limite en ligne
        for y in range(GRID_SIZE):
            count += 1

    return count

def nombre_facons_liste_bateaux(grille, liste_bateaux):
    """
    Calcule le nombre total de façons de placer une liste de bateaux sur une grille vide.
    """
    total_facons = 1

    for bateau in liste_bateaux:
        facons_bateau = nombre_de_facons_placer(grille, bateau)
        total_facons *= facons_bateau

    return total_facons
# sur des listes de 1, 2 et 3 bateaux
grille_vide = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Liste d'un seul bateau : porte-avions (id = 1)
liste_un_bateau = [1]  # Porte-avions
nombre_un_bateau = nombre_facons_liste_bateaux(grille_vide, liste_un_bateau)
print(f"Nombre de façons de placer {liste_un_bateau}: {nombre_un_bateau}")

# Liste de deux bateaux : porte-avions (id = 1) et croiseur (id = 2)
liste_deux_bateaux = [1, 2]  # Porte-avions et Croiseur
nombre_deux_bateaux = nombre_facons_liste_bateaux(grille_vide, liste_deux_bateaux)
print(f"Nombre de façons de placer {liste_deux_bateaux}: {nombre_deux_bateaux}")

# Liste de trois bateaux : porte-avions (id = 1), croiseur (id = 2), contre-torpilleur (id = 3)
liste_trois_bateaux = [1, 2, 3]  # Porte-avions, Croiseur et Contre-torpilleur
nombre_trois_bateaux = nombre_facons_liste_bateaux(grille_vide, liste_trois_bateaux)
print(f"Nombre de façons de placer {liste_trois_bateaux}: {nombre_trois_bateaux}")

######
# 2.4 Le nombre de répétitions avant d’obtenir deux grilles égales

import numpy as np

def eq(grilleA, grilleB):
    """
    Compare deux grilles pour vérifier si elles sont égales.
    """
    return np.array_equal(grilleA, grilleB)

def genere_grille():
    """
    Génère une grille comprenant tous les bateaux disposés de manière aléatoire.
    """
    grille = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for bateau in SHIP_IDS.values():
        grille = place_alea(grille, bateau)

    return grille

def cherche_grille_identique(grille_cible):
    """
    Génère des grilles aléatoirement jusqu'à trouver une grille identique à la grille donnée.
    Renvoie le nombre de grilles générées avant de trouver la bonne.
    """
    compteur = 0
    while True:
        compteur += 1
        grille_generee = genere_grille()
        if eq(grille_generee, grille_cible):
            break

    return compteur

# Générons une grille cible aléatoire
# grille_cible = genere_grille()

# Appelons la fonction pour trouver combien de grilles sont générées avant de trouver la même
# nombre_tentatives = cherche_grille_identique(grille_cible)
# print(f"Nombre de grilles générées avant de trouver la grille cible : {nombre_tentatives}")

######
# 2.5 Approximation

import numpy as np

# Paramètres du jeu
GRID_SIZE = 10
SHIP_IDS = {1: "Porte-avions", 2: "Croiseur", 3: "Contre-torpilleur", 4: "Sous-marin", 5: "Torpilleur"}
SHIP_LENGTHS = {1: 5, 2: 4, 3: 3, 4: 3, 5: 2}

def nombre_de_facons_placer_bateau(grille, bateau_length):
    """
    Calcule le nombre total de façons de placer un bateau sur une grille donnée (partiellement remplie ou vide).
    """
    count = 0
    # Cas de placement horizontal (direction = 1)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE - bateau_length + 1):  # Limite en colonne pour placement horizontal
            if np.all(grille[x, y:y + bateau_length] == 0):  # Vérifie si l'espace est libre
                count += 1

    # Cas de placement vertical (direction = 2)
    for x in range(GRID_SIZE - bateau_length + 1):  # Limite en ligne pour placement vertical
        for y in range(GRID_SIZE):
            if np.all(grille[x:x + bateau_length, y] == 0):  # Vérifie si l'espace est libre
                count += 1

    return count

def approximer_nombre_total_grilles(liste_bateaux):
    """
    Approximativement calcule le nombre total de grilles possibles pour une liste de bateaux.
    En réduisant l'espace disponible à chaque placement de bateau.
    """
    grille = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    total_facons = 1

    for bateau in liste_bateaux:
        longueur_bateau = SHIP_LENGTHS[bateau]
        facons_bateau = nombre_de_facons_placer_bateau(grille, longueur_bateau)
        if facons_bateau == 0:
            return 0  # Si aucune façon de placer le bateau, stop
        total_facons *= facons_bateau
        
        # Pour simplifier, plaçons un bateau aléatoirement dans la grille pour réduire l'espace disponible
        place_alea(grille, bateau)  # Cette fonction place le bateau de manière aléatoire dans la grille

    return total_facons

######

def place_alea(grille, bateau):
    """
    Place un bateau aléatoirement sur la grille sans se soucier des autres bateaux.
    """
    longueur_bateau = SHIP_LENGTHS[bateau]
    placed = False
    while not placed:
        # Tirage aléatoire d'une position et d'une direction
        direction = np.random.choice([1, 2])  # 1: Horizontal, 2: Vertical
        x, y = np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)

        if direction == 1 and y + longueur_bateau <= GRID_SIZE:
            if np.all(grille[x, y:y + longueur_bateau] == 0):  # Vérifie si l'espace est libre
                grille[x, y:y + longueur_bateau] = bateau
                placed = True

        elif direction == 2 and x + longueur_bateau <= GRID_SIZE:
            if np.all(grille[x:x + longueur_bateau, y] == 0):  # Vérifie si l'espace est libre
                grille[x:x + longueur_bateau, y] = bateau
                placed = True

# liste_bateaux = [1, 2, 3, 4, 5]  # Tous les bateaux du jeu
# nombre_approximatif = approximer_nombre_total_grilles(liste_bateaux)
# print(f"Nombre approximatif de grilles possibles : {nombre_approximatif}")




###### 4 - Senseur imparfait
# 4.5 - Algorithme de recherche 


import numpy as np

def bayesian_search(N, ps, max_iterations=100, tolerance=1e-4):
    # Initial prior probabilities (uniform prior)
    pi = np.ones(N) / N
    
    # Perform iterations
    for iteration in range(1, max_iterations + 1):
        previous_pi = pi.copy()
        
        # Perform prediction step
        for i in range(N):
            # Update prior probabilities using Bayes' theorem
            pi[i] = (previous_pi[i] * ps) / (np.sum(previous_pi * ps))
        
        # Simulate search results (here, we assume a simple detection simulation)
        # In a real application, you would integrate your detection model here.
        detection_results = np.random.rand(N) < pi
        
        # Check convergence
        if np.linalg.norm(pi - previous_pi) < tolerance:
            break
    
    return pi

N = 100  # Grid size (for a 5x5 grid)
ps = 0.8  # Probability of successful detection if the object is present
pi_estimated = bayesian_search(N, ps)

print("Estimated probabilities after Bayesian search:")
print(pi_estimated.reshape((int(np.sqrt(N)), int(np.sqrt(N)))))




######
# 4.6 - Tests de l’algorithme de recherche d’objet


import numpy as np

def update_probabilities(pi, ps, k):
    """
    Met à jour les probabilités après l'échec de la détection dans la case k.
    pi : tableau des probabilités actuelles
    ps : probabilité de détection
    k  : indice de la case scannée
    """
    # Mise à jour de la probabilité de la case scannée k
    pi[k] = pi[k] * (1 - ps) / (1 - pi[k] * ps)
    
    # Mise à jour des autres probabilités
    for i in range(len(pi)):
        if i != k:
            pi[i] = pi[i] / (1 - pi[k] * ps)
    
    return pi

def bayesian_search(N, ps, max_iterations=200):
    """
    Recherche bayésienne d'un objet dans une grille de N cases avec probabilité de détection ps.
    N : nombre total de cases
    ps : probabilité de succès de détection si l'objet est dans la case scannée
    max_iterations : nombre maximum d'itérations
    """
    # Initialisation des probabilités a priori (uniforme)
    pi = np.ones(N) / N
    
    # Simulation de la position réelle de l'objet
    true_position = np.random.randint(0, N)  # L'objet est dans une case aléatoire
    
    # Itérations de recherche
    for iteration in range(1, max_iterations + 1):
        # Sélectionner la case avec la plus grande probabilité
        k = np.argmax(pi)
        
        # Scanner la case k
        print(f"Iteration {iteration}: Scanning cell {k} with probability {pi[k]:.4f}")
        
        # Simulation de la détection
        if k == true_position:
            detection = np.random.rand() < ps  # Succès de détection avec probabilité ps
        else:
            detection = False  # Échec de la détection si ce n'est pas la bonne case
        
        if detection:
            print(f"Object found in cell {k} after {iteration} iterations!")
            return k, iteration  # L'objet est trouvé, retourner la position et le nombre d'itérations
        else:
            # Mise à jour des probabilités après échec de la détection
            pi = update_probabilities(pi, ps, k)
    
    print("Object not found within the maximum iterations.")
    return None, max_iterations

N = 100  # Taille de la grille (par exemple une grille 10x10)
ps = 0.8  # Probabilité de succès de détection si l'objet est présent
position_trouvee, iterations = bayesian_search(N, ps)

if position_trouvee is not None:
    print(f"L'objet a été trouvé à la position {position_trouvee} après {iterations} itérations.")
else:
    print(f"L'objet n'a pas été trouvé après {iterations} itérations.")

import numpy as np
import matplotlib.pyplot as plt

def update_probabilities(pi, ps, k):
    """
    Met à jour les probabilités après l'échec de la détection dans la case k.
    pi : tableau des probabilités actuelles
    ps : probabilité de détection
    k  : indice de la case scannée
    """
    # Mise à jour de la probabilité de la case scannée k
    pi[k] = pi[k] * (1 - ps) / (1 - pi[k] * ps)
    
    # Mise à jour des autres probabilités
    for i in range(len(pi)):
        if i != k:
            pi[i] = pi[i] / (1 - pi[k] * ps)
    
    return pi

def bayesian_search(N, ps, max_iterations=100):
    """
    Recherche bayésienne d'un objet dans une grille de N cases avec probabilité de détection ps.
    N : nombre total de cases
    ps : probabilité de succès de détection si l'objet est dans la case scannée
    max_iterations : nombre maximum d'itérations
    """
    # Initialisation des probabilités a priori (uniforme)
    pi = np.ones(N) / N
    
    # Simulation de la position réelle de l'objet
    true_position = np.random.randint(0, N)  # L'objet est dans une case aléatoire
    
    # Itérations de recherche
    for iteration in range(1, max_iterations + 1):
        # Sélectionner la case avec la plus grande probabilité
        k = np.argmax(pi)
        
        # Simulation de la détection
        if k == true_position:
            detection = np.random.rand() < ps  # Succès de détection avec probabilité ps
        else:
            detection = False  # Échec de la détection si ce n'est pas la bonne case
        
        if detection:
            return iteration  # L'objet est trouvé, retourner le nombre d'itérations
        else:
            # Mise à jour des probabilités après échec de la détection
            pi = update_probabilities(pi, ps, k)
    
    return max_iterations  # Si l'objet n'est pas trouvé, retourner le nombre max d'itérations

# Configuration de la simulation
N = 100  # Taille de la grille
ps = 0.8  # Probabilité de succès de détection si l'objet est présent
max_iterations = 1000  # Nombre maximum d'itérations
num_searches = 1000  # Nombre de recherches à effectuer

# Exécuter 1000 recherches
iterations_needed = [bayesian_search(N, ps, max_iterations) for _ in range(num_searches)]

# Tracer l'histogramme du nombre d'itérations nécessaires
plt.figure(figsize=(10, 6))
plt.hist(iterations_needed, bins=range(1, max_iterations + 1), edgecolor='black', alpha=0.7)
plt.title('Distribution du nombre d\'itérations nécessaires pour 1000 recherches')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def update_probabilities(pi, ps, k):
    """
    Met à jour les probabilités après l'échec de la détection dans la case k.
    pi : tableau des probabilités actuelles
    ps : probabilité de détection
    k  : indice de la case scannée
    """
    # Mise à jour de la probabilité de la case scannée k
    pi[k] = pi[k] * (1 - ps) / (1 - pi[k] * ps)
    
    # Mise à jour des autres probabilités
    for i in range(len(pi)):
        if i != k:
            pi[i] = pi[i] / (1 - pi[k] * ps)
    
    return pi

def bayesian_search_with_priors(N, ps, max_iterations=100, prior_distribution="uniform"):
    """
    Recherche bayésienne d'un objet dans une grille de N cases avec probabilité de détection ps.
    Utilise différentes distributions a priori.
    N : nombre total de cases
    ps : probabilité de succès de détection si l'objet est dans la case scannée
    max_iterations : nombre maximum d'itérations
    prior_distribution : type de distribution a priori (uniform, centered, border)
    """
    # Initialisation des probabilités a priori selon la distribution choisie
    if prior_distribution == "uniform":
        pi = np.ones(N) / N  # Distribution uniforme
    elif prior_distribution == "centered":
        grid_size = int(np.sqrt(N))  # Assuming square grid
        center = (grid_size // 2, grid_size // 2)
        pi = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                pi[i, j] = np.exp(-distance)  # Higher probability near the center
        pi = pi.flatten() / pi.sum()  # Normalize to get probabilities
    elif prior_distribution == "border":
        grid_size = int(np.sqrt(N))  # Assuming square grid
        pi = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
                    pi[i, j] = 1  # Higher probability on borders
                else:
                    pi[i, j] = 0.1  # Lower probability in the center
        pi = pi.flatten() / pi.sum()  # Normalize to get probabilities
    
    # Simulation de la position réelle de l'objet
    true_position = np.random.randint(0, N)  # L'objet est dans une case aléatoire
    
    # Itérations de recherche
    for iteration in range(1, max_iterations + 1):
        # Sélectionner la case avec la plus grande probabilité
        k = np.argmax(pi)
        
        # Simulation de la détection
        if k == true_position:
            detection = np.random.rand() < ps  # Succès de détection avec probabilité ps
        else:
            detection = False  # Échec de la détection si ce n'est pas la bonne case
        
        if detection:
            return iteration  # L'objet est trouvé, retourner le nombre d'itérations
        else:
            # Mise à jour des probabilités après échec de la détection
            pi = update_probabilities(pi, ps, k)
    
    return max_iterations  # Si l'objet n'est pas trouvé, retourner le nombre max d'itérations

# Configuration de la simulation
N = 100  # Taille de la grille (10x10)
ps = 0.8  # Probabilité de succès de détection si l'objet est présent
max_iterations = 1000  # Nombre maximum d'itérations
num_searches = 1000  # Nombre de recherches à effectuer

# Effectuer les recherches pour chaque distribution
distributions = ['uniform', 'centered', 'border']
results = {}

for distribution in distributions:
    iterations_needed = [bayesian_search_with_priors(N, ps, max_iterations, distribution) for _ in range(num_searches)]
    results[distribution] = iterations_needed

# Tracer les histogrammes comparatifs
plt.figure(figsize=(12, 8))

for i, distribution in enumerate(distributions):
    plt.subplot(3, 1, i + 1)
    plt.hist(results[distribution], bins=range(1, max_iterations + 1), edgecolor='black', alpha=0.7)
    plt.title(f'Distribution du nombre d\'itérations nécessaires - {distribution.capitalize()} prior')
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Fréquence')
    plt.grid(True)

plt.tight_layout()
plt.show()
