<div align="center">

# <span style="color: #2c7be5;">PyTorch CNN Bildklassifizierungsprojekt 🚀🧠</span>

[简体中文](README_CN.md) / [繁体中文](README_TC.md) / [English](README.md) / Deutsch / [日本語](README_JP.md)

</div>

Dies ist ein CNN-Bildklassifizierungsprojekt, das mit dem PyTorch-Framework implementiert wurde und einen vollständigen Trainingsablauf sowie Datenverarbeitungsfunktionen bietet. Das Projekt integriert einen Aufmerksamkeitsmechanismus, unterstützt mehrere Datenverstärkungsmethoden und bietet einen vollständigen Trainings- und Evaluierungsablauf.

## <span style="color: #228be6;">Projektstruktur 📁🗂️</span>

```
├── models/             # Modellbezogene Definitionen
│   ├── cnn.py         # Grundlegendes CNN-Modell
│   └── attention.py   # Aufmerksamkeitsmechanismus-Modul
├── data_processing/    # Datenverarbeitungsbezogen
│   └── dataset.py     # Datensatzladen und Vorverarbeitung
├── trainers/          # Trainingsbezogen
│   └── trainer.py     # Trainer-Implementierung
├── utils/             # Hilfsfunktionen
│   ├── config.py      # Konfigurationsverwaltung
│   └── visualization.py # Visualisierungstools
├── tests/             # Testcode
│   └── test_model.py  # Modelltest
├── static/            # Statische Ressourcen
├── templates/         # Web-Vorlagen
├── predict.py         # Vorhersageskript
├── main.py           # Haupteinstiegspunkt des Programms
└── requirements.txt   # Projektabhängigkeiten
```

## <span style="color: #228be6;">Hauptmerkmale ✨🌟</span>

<span style="color: #38d9a9;">- Integrierter Aufmerksamkeitsmechanismus zur Verbesserung der Modellleistung</span>
<span style="color: #38d9a9;">- Unterstützung mehrerer Datenverstärkungsmethoden</span>
<span style="color: #38d9a9;">- Bereitstellung einer Web-Oberfläche für Online-Vorhersagen</span>
<span style="color: #38d9a9;">- Unterstützung der Visualisierung des Modelltrainingsablaufs</span>
<span style="color: #38d9a9;">- Vollständige Testfälle</span>

## <span style="color: #228be6;">Umgebungsconfiguration ⚙️🛠️</span>

1. Erstellen und Aktivieren einer virtuellen Umgebung (empfohlen):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Installieren der Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

3. CUDA-Unterstützung (empfohlen):
   - Stellen Sie sicher, dass der NVIDIA GPU-Treiber installiert ist.
   - PyTorch erkennt automatisch und nutzt verfügbare GPUs.

## <span style="color: #228be6;">Datenvorbereitung 📊📂</span>

1. Organisieren Sie den Datensatz im `data`-Verzeichnis:
   - Erstellen Sie für jede Kategorie einen Unterordner.
   - Legen Sie die Bilder der entsprechenden Kategorie in den jeweiligen Ordner.

Beispielstruktur:
```
data/
  ├── 淡白舌灰黑苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  ├── 淡白舌白苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  └── ...
```

## <span style="color: #228be6;">Modelltraining 🧠💪</span>

1. Konfigurieren Sie die Trainierungsparameter.
   Bearbeiten Sie die Datei `utils/config.py`:
   ```python
   # Setzen Sie die Trainierungsparameter
   num_classes = 15  # Anzahl der Klassifikationskategorien
   batch_size = 32   # Batch-Größe
   num_epochs = 100  # Anzahl der Trainings-Epochen
   learning_rate = 0.001  # Lernrate
   ```

2. Starten Sie das Training:
   ```bash
   python main.py --mode train
   ```

3. Visualisierung des Trainingsablaufs:
   - Die Verlustkurven und Genauigkeitskurven werden in Echtzeit aktualisiert.
   - Modell-Checkpoints werden automatisch im `checkpoints`-Verzeichnis gespeichert.

## <span style="color: #228be6;">Verwenden des Modells zur Vorhersage 🎯✅</span>

### Web-Oberflächenvorhersage

1. Starten Sie den Web-Dienst:
   ```bash
   python app.py
   ```

2. Besuchen Sie `http://localhost:5000` für Online-Vorhersagen.

### Befehlszeilenvorhersage

```python
from predict import ImagePredictor

# Initialisieren Sie den Vorhersager
predictor = ImagePredictor('checkpoints/best_model.pth')

# Einzelbildvorhersage
result = predictor.predict_single('data/ak47/001_0001.jpg')
print(f'Vorhergesagte Klasse: {result["class"]}')
print(f'Konfidenz: {result["probability"]}')

# Batch-Vorhersage
results = predictor.predict_batch('data/ak47')
for result in results:
    print(f'Bild: {result["image"]}')
    print(f'Vorhersageergebnis: {result["prediction"]}')
```

## <span style="color: #228be6;">Modellarchitektur 🏗️</span>

- Grundlegende CNN-Architektur: 3 Convolutional Layer-Blöcke (Convolution + Batch-Normalisierung + ReLU + Pooling)
- Aufmerksamkeitsmechanismus: Selbst-Aufmerksamkeitsmodul zur Verstärkung der Merkmalsextraktion
- Vollständig verbundene Schichten: 3 Schichten zur Merkmalsreduzierung und Klassifikation
- Dropout-Schicht: Vermeidung von Überanpassung
- Verlustfunktion: Kreuzentropieverlust
- Optimierer: Adam

## <span style="color: #228be6;">Hinweise ⚠️</span>

- Unterstützte Bildformate: jpg, jpeg, png
- Empfohlen wird die Verwendung einer GPU für das Training.
- Das Modell kann durch Bearbeiten der Konfigurationsdatei angepasst werden.
- Machen Sie regelmäßig Backups der trainierten Modelldateien.
- Stellen Sie sicher, dass der Pfad zur Modelldatei bei der Vorhersage korrekt ist.