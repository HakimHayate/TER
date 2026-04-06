import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds 
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def creer_et_entrainer_modele():
    print("📥 Téléchargement sécurisé du dataset EMNIST (Digits) via TensorFlow Datasets...")
    
    train_data = tfds.load('emnist/digits', split='train', batch_size=-1, as_supervised=True)
    test_data = tfds.load('emnist/digits', split='test', batch_size=-1, as_supervised=True)
    
    # Extraction des images et des labels
    train_data = tfds.as_numpy(train_data)
    test_data = tfds.as_numpy(test_data)
    
    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    print("🔄 Remise à l'endroit des images et normalisation...")
    x_train = np.transpose(x_train, axes=(0, 2, 1, 3))
    x_test = np.transpose(x_test, axes=(0, 2, 1, 3))

    # Normalisation classique entre 0 et 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"📊 Taille du dataset d'entraînement : {x_train.shape[0]} images prêtes !")

    print("🧠 Construction du Réseau de Neurones (CNN)...")
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1, input_shape=(28, 28, 1)), 
        layers.RandomZoom(0.1),                              
        layers.RandomTranslation(0.1, 0.1),                  
    ])

    model = models.Sequential([
        data_augmentation,

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1), 

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), 

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3), 

        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4), 

        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ralentisseur = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, min_lr=0.00001)
    arret_urgence = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    print("\n🚀 Début de l'entraînement massif sur EMNIST avec Augmentation et Callbacks...")
    
    history = model.fit(
        x_train, y_train, 
        epochs=20, 
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[ralentisseur, arret_urgence]
    )
    
    print("\n💾 Sauvegarde du nouveau modèle...")
    model.save("modele_emnist.h5")
    print("✅ Terminé !")

if __name__ == "__main__":
    creer_et_entrainer_modele()