import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model(r'C:\Users\Caisse\OneDrive\Desktop\rapportpfecode\ELM+CNN.ipynb')  # Remplacez par le chemin de votre modèle

def preprocess_image(image):
    # Prétraiter l'image selon les besoins de votre modèle
    # ...
    # Par exemple, pour un modèle pré-entraîné sur des images de taille 256x256:
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalisation des valeurs de pixel
    image = np.expand_dims(image, axis=0)  # Ajout d'une dimension supplémentaire pour correspondre à la forme d'entrée du modèle
    return image

def classify_image(file_path):
    # Charger l'image
    image = Image.open(file_path)

    # Prétraitement de l'image
    processed_image = preprocess_image(image)

    # Effectuer la prédiction
    prediction = model.predict(processed_image)[0]
    class_label = 'M' if prediction >= 0.5 else 'B'

    # Afficher la prédiction
    print(f"Prédiction: {class_label} avec une probabilité de {prediction:.2%}")

# Interface utilisateur Tkinter
def on_choose_button_click():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Charger l'image
            image = Image.open(file_path)
            image = image.resize((250, 250))  # Ajustez la taille selon les besoins

            # Afficher l'image dans l'interface
            photo = ImageTk.PhotoImage(image)
            canvas.itemconfig(canvas_image, image=photo)
            canvas.image = photo

            # Classer l'image
            classify_image(file_path)
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