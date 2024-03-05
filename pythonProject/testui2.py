import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model(r'D:\S6\IHM\projet\CNN_model.h5')  # Remplacez par le chemin de votre modèle

# Définir une variable globale pour stocker le label prédit
predicted_label = None

def preprocess_image(image):
    # Prétraiter l'image selon les besoins de votre modèle
    # Convertir l'image en mode RGB
    image = image.convert('RGB')
    # Redimensionner et ajouter une dimension pour les canaux de couleur (RGB)
    image = image.resize((50, 50))
    image = np.array(image) / 255.0  # Normalisation des valeurs de pixel
    image = np.expand_dims(image, axis=0)  # Ajout d'une dimension supplémentaire pour correspondre à la forme d'entrée du modèle
    return image

def classify_image(file_path):
    global predicted_label
    # Charger l'image
    image = Image.open(file_path)

    # Prétraitement de l'image
    processed_image = preprocess_image(image)

    # Effectuer la prédiction
    prediction = model.predict(processed_image)[0]
    if prediction[0] >= 0.5:
        predicted_label = 'M'
    else:
        predicted_label = 'B'

    # Afficher la prédiction
    print(f"Prédiction: {predicted_label} avec une probabilité de {prediction[0]:.2%}")



# Interface utilisateur Tkinter
def on_choose_button_click():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Charger l'image
            image = Image.open(file_path)
            image = image.resize((50, 50))  # Ajustez la taille selon les besoins

            # Afficher l'image dans l'interface
            photo = ImageTk.PhotoImage(image)
            canvas.itemconfig(canvas_image, image=photo)
            canvas.image = photo

            # Classer l'image
            classify_image(file_path)
            if predicted_label is not None:
                print("Label prédit à l'extérieur de la fonction:", predicted_label)
        except Exception as e:
            print("Erreur lors du chargement de l'image ou de la prédiction:", str(e))

# Créer la fenêtre principale
root = tk.Tk()
root.title("Détection de cancer du sein")

# Bouton pour choisir un fichier
choose_button = tk.Button(root, text="Choisir une image", command=on_choose_button_click)
choose_button.pack(pady=10)

# Canvas pour afficher l'image
canvas = tk.Canvas(root, width=250, height=250)
canvas.pack()

# Placeholder pour l'image
canvas_image = canvas.create_image(0, 0, anchor=tk.NW)

# Exécution de l'interface
root.mainloop()
