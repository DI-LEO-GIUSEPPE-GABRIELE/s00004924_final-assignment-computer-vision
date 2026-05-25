# Appunti Didattici (IT) — Medical Image Analysis Tool (Segmentation + Classification)

Questo file è pensato solo per studio/esposizione orale. Spiega il progetto “da zero”: concetti, pipeline, file, comandi, scelte tecniche e una comparativa 1:1 con la comanda (requirements dell’esame).

---

## 1) Obiettivo del progetto (in parole semplici)

Vogliamo costruire una piccola applicazione di Computer Vision che, data un’immagine medica (nel nostro caso una chest X-ray/CXR):

Una **chest X-ray** (abbreviata **CXR**) è una **radiografia del torace**.

- **Che cos’è**: un esame di imaging in cui un fascio di raggi X attraversa il torace e viene registrato da un sensore, producendo un’immagine 2D.
- **Cosa si vede tipicamente**: polmoni, cuore, coste, diaframma e (in parte) la colonna vertebrale.
- **Perché è utile**: è uno degli esami più comuni e rapidi per valutare problemi toracici (es. infezioni, versamenti, alterazioni polmonari).
- **Perché nel nostro progetto**: è un tipo di immagine standard in ambito medico e spesso usata per task di segmentazione (es. polmoni) e classificazione (es. normal vs abnormal).

1. **Segmenta** una regione di interesse (ROI) creando una **maschera binaria** pixel-per-pixel (1 = ROI, 0 = background).
   - Nel dataset reale usato qui, la ROI è il **campo polmonare** (lung fields).
2. **Classifica** l’immagine con un’etichetta binaria **0/1** (per esempio: normal vs abnormal legato alla TB).

Il valore aggiunto è che l’output non è solo un “numero”, ma anche una maschera che rende più spiegabile il risultato (una forma di interpretabilità visuale).

---

## 2) Concetti base (per chi parte da zero)

### 2.1 Immagine digitale e pixel

- Un’immagine è una griglia di **pixel**.
- In grayscale ogni pixel ha intensità 0–255 (uint8).  
  0 = nero, 255 = bianco.

### 2.2 Segmentazione (segmentation)

Segmentazione = assegnare una classe ad ogni pixel. Qui è **binaria**:

- 1: pixel appartenente ai polmoni (ROI)
- 0: background

Output tipico di un modello: una “mappa” H×W con valori continui (probabilità o logits), poi convertiti in binario con una soglia.

### 2.3 Classificazione (classification)

Classificazione = assegnare una classe all’immagine intera. Qui è **binaria**:

- 0: normal
- 1: abnormal (TB)

Output tipico di un modello: un singolo numero (logit) che convertiamo in probabilità con sigmoid.

### 2.4 Logits, Sigmoid e soglia

Molti modelli non outputtano direttamente probabilità ma **logits** (numeri reali).

- Applichi **sigmoid(logit)** per ottenere una probabilità in [0,1].
- Applichi una soglia (es. 0.5) per decidere classe 0/1 o maschera binaria.

### 2.5 Preprocessing (prima del modello)

Serve per rendere i dati più “puliti” e uniformi:

- denoise (riduzione rumore)
- resize (dimensione fissa)
- normalization (stabilizza la distribuzione dei pixel)

### 2.6 Data augmentation (solo in training)

Sono trasformazioni che simulano variabilità reale per ridurre overfitting:

- flip orizzontale/verticale
- rotazioni a multipli di 90°

Importante: per segmentation, l’augment deve essere **coerente** su immagine e maschera (stessa trasformazione).

### 2.7 Post-processing (dopo il modello)

Serve per “ripulire” la maschera finale:

- threshold: probabilità → binario
- morfologia (opening/closing) per rimuovere puntini e riempire buchi
- largest connected component: tenere solo il componente più grande (utile quando compaiono frammenti spuri)

### 2.8 Che cos’è il Machine Learning (ML) e perché “impara”

Computer Vision “classica” = regole scritte a mano (es. soglie, edge detection, ecc.).  
Deep Learning = invece di scrivere regole, fai vedere esempi e il modello impara i parametri che minimizzano un errore.

Concetti base:

- Dati: input (immagini) + target (maschere, label).
- Modello: una funzione parametrica (qui una rete neurale) che produce output.
- Loss: misura quanto l’output del modello è sbagliato rispetto al target.
- Ottimizzazione: aggiorna i parametri per ridurre la loss.

### 2.9 Supervised learning (apprendimento supervisionato)

Qui siamo in supervised learning perché abbiamo ground truth:

- per segmentation: la maschera reale (annotazione)
- per classification: etichetta 0/1

Il training consiste nel trovare pesi che generalizzano: non devono memorizzare solo il training, ma funzionare bene su dati nuovi.

### 2.10 Train/Validation/Test (perché 3 split)

- Train: usato per aggiornare i pesi.
- Validation (val): usato per scegliere iperparametri e selezionare il checkpoint migliore (senza barare sul test).
- Test: usato per stimare performance finale realistica.

Se usassi il test per scegliere i parametri, faresti data leakage e gonfieresti artificialmente i risultati.

### 2.11 Overfitting vs generalization

- Overfitting: il modello va bene sul train ma male su val/test (ha memorizzato).
- Generalization: performance simile su train e val/test.

Tecniche anti-overfitting che puoi citare:

- data augmentation
- weight decay (AdamW)
- dropout (qui usato nel classification head)
- early stopping (non implementato, ma il best checkpoint è una forma di selezione)

### 2.12 Ottimizzazione: epoch, batch size, learning rate (iperparametri)

- Epoch: un passaggio completo su tutto il train set.
- Batch: un sottoinsieme del train usato per un aggiornamento dei pesi.
- Batch size: quanti esempi per batch (trade-off: stabilità gradiente vs memoria).
- Learning rate (LR): quanto muovi i pesi ad ogni update. Troppo alto = divergenza, troppo basso = training lento.

### 2.13 Gradient descent e backprop (versione orale)

Idea semplice:

1. Il modello produce output.
2. Calcoli la loss.
3. Calcoli come cambia la loss se cambi ogni peso (gradiente).
4. Aggiorni i pesi nella direzione che riduce la loss.

Questa procedura si chiama backpropagation.

---

## 3) Architettura del modello (Multi-task UNet)

### 3.1 Perché UNet

UNet è un’architettura molto usata in medical imaging per segmentation:

- ha un **encoder** che comprime l’immagine (estrae feature)
- ha un **decoder** che ricostruisce la maschera alla risoluzione originale
- usa **skip connections**: passano informazioni di dettaglio dall’encoder al decoder

### 3.2 Multi-task learning (seg+cls insieme)

Invece di addestrare due modelli separati, usiamo un solo backbone (encoder condiviso) e due “teste”:

- testa segmentation: produce H×W logits
- testa classification: usa le feature del bottleneck (global pooling) e produce 1 logit

Vantaggi:

- condivisione delle feature
- più efficiente
- classification beneficia di rappresentazioni anatomiche (lungs) e viceversa

### 3.3 File del modello

- Modello: [mtl_unet.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/models/mtl_unet.py)
  - `ConvBlock`: blocchi conv+BN+ReLU
  - `UpBlock`: upsample + concatenazione skip + conv
  - `MultiTaskUNet`: rete completa con `seg_head` e `cls_head`

Pattern tipico: “Encoder-Decoder + Skip connections” (UNet) + “Multi-head” (multi-task).

### 3.4 CNN da zero: convoluzione, feature maps, stride, padding

Per argomentare bene, ecco le basi:

- Una convoluzione applica un filtro (kernel, es. 3×3) che scorre sull’immagine e produce una feature map.
- Ogni filtro impara pattern: bordi, texture, poi strutture più complesse.
- Stride: passo dello scorrimento. Stride=2 dimezza la risoluzione.
- Padding: aggiunge bordi per non perdere dimensione e preservare informazioni ai margini.
- Una CNN condivide pesi nello spazio: riconosce pattern ovunque nell’immagine.

### 3.5 Pooling e perché si riduce la risoluzione

- MaxPool (o stride conv) riduce H×W: aumenta il campo recettivo (receptive field) e rende le feature più robuste.
- In UNet, ridurre e poi risalire permette di combinare:
  - contesto globale (bottleneck)
  - dettaglio locale (skip connections)

### 3.6 BatchNorm e Dropout (perché ci sono)

- BatchNorm stabilizza la distribuzione interna delle attivazioni, rendendo training più stabile.
- Dropout spegne casualmente neuroni durante training, riducendo overfitting.
  - Qui è nella cls_head (classification).

---

## 4) Loss function (come il modello impara)

### 4.1 Loss per classification

Usiamo **BCEWithLogitsLoss**:

- BCE = Binary Cross-Entropy
  Intuizione:
- se y=1 vuoi p vicino a 1
- se y=0 vuoi p vicino a 0
  La BCE penalizza molto quando sei “sicuro ma sbagliato”.

- “WithLogits” significa che accetta logits e applica internamente sigmoid in modo numericamente stabile

### 4.2 Loss per segmentation

Perché due loss?

- BCE: penalizza pixel sbagliati
- Dice: ottimizza la sovrapposizione maschera-predizione, utile con class imbalance (ROI piccola rispetto al background)

### 4.3 Loss totale multi-task

Somma pesata:
`total = w_cls * cls_bce + w_seg_bce * seg_bce + w_dice * dice`

File: [losses.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/losses.py)

### 4.4 Perché Dice è utile in medical imaging (class imbalance)

Spesso la ROI è piccola rispetto al background:

- BCE può essere “ingannata” se il modello predice quasi tutto background (tanti pixel corretti).
- Dice guarda la sovrapposizione della ROI e spinge a segmentare bene la parte minoritaria.

---

## 5) Metriche di valutazione (evaluation)

### 5.1 Segmentation

1. **Dice coefficient**
   - misura sovrapposizione: 2|A∩B| / (|A|+|B|)
   - va da 0 (male) a 1 (perfetto)
2. **IoU / mIoU**
   - IoU = |A∩B| / |A∪B|
   - nel binario, “mIoU” è la media su batch

File: [metrics.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/metrics.py)

### 5.2 Classification

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion Matrix (2×2)

Queste metriche vengono calcolate in evaluation con scikit-learn.

File evaluation loop: [engine.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/engine.py)

### 5.3 Confusion Matrix spiegata bene (TP/TN/FP/FN)

Per una classificazione binaria:

- TP (True Positive): predici 1 e la verità è 1
- TN (True Negative): predici 0 e la verità è 0
- FP (False Positive): predici 1 ma la verità è 0
- FN (False Negative): predici 0 ma la verità è 1

Da questi derivano:

- Precision = TP / (TP + FP) → tra i positivi predetti, quanti sono corretti
- Recall = TP / (TP + FN) → tra i positivi veri, quanti ne trovi
- F1 = 2 _ (Precision _ Recall) / (Precision + Recall) → compromesso tra precision e recall

Nota orale (ambito medico):

- I FN possono essere critici (casi persi).
- I FP aumentano carico e allarmi inutili.

### 5.4 Soglia di decisione (0.5 non è obbligatorio)

Per classification e segmentation usiamo una soglia (default 0.5).
All’orale puoi dire che spesso si sceglie la soglia su validation:

- per massimizzare F1
- per controllare FN (alta recall) se il costo clinico è alto
- per bilanciare precision/recall

### 5.5 Perché “Accuracy” da sola può essere ingannevole

Se il dataset è sbilanciato (molti 0, pochi 1):

- un modello che predice sempre 0 può avere accuracy alta ma è inutile
  Per questo si usano anche precision/recall/F1 e confusion matrix.

---

## 6) Post-processing (segmentation)

File: [postprocess.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/postprocess.py)

### 6.1 Operazioni morfologiche (concetto)

- **Opening** = erosione poi dilatazione: rimuove piccoli “puntini” rumorosi
- **Closing** = dilatazione poi erosione: chiude piccoli buchi nella ROI

### 6.2 Largest connected component

Quando la maschera ha frammenti, teniamo solo il componente con area maggiore.

### 6.3 Perché il post-processing è importante (e perché è richiesto)

La comanda chiede esplicitamente un post-processing “per rifinire i risultati”.
In segmentation è molto comune perché:

- riduci falsi positivi piccoli e isolati
- riempi buchi interni alla ROI
- ottieni maschere più stabili e “anatomiche”

---

## 7) Dataset

### 7.1 Formato atteso dal progetto (pattern)

Per ogni split:

- `images/*.png`
- `masks/*.png`
- `labels.csv` con colonne `id,label`

La classe dataset carica immagine + maschera + label usando lo stesso `id`.

File dataset: [data.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/data.py)

### 7.2 Dataset reale: Montgomery County CXR (NLM)

Scaricato e convertito nel formato del progetto con:

- Script: [prepare_montgomery.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/prepare_montgomery.py)

#### 7.2.1 Da dove ho recuperato il dataset (risposta da orale)

Ho usato il **Montgomery County Chest X-ray (CXR) dataset** rilasciato pubblicamente dalla **U.S. National Library of Medicine (NLM)** (Lister Hill National Center for Biomedical Communications).  
Link alla pagina indice dei file (immagini + maschere):  
https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html

Dettaglio importante (da citare se serve):

- include CXR de-identificate
- include maschere manuali dei polmoni (left/right) che lo script unisce in una maschera unica
- la label 0/1 è codificata nel nome file (`_0` normal, `_1` abnormal)

Che cosa fa lo script:

- scarica immagini CXR e maschere polmoni left/right
- unisce left+right in una maschera unica
- crea train/val/test stratificato
- genera `labels.csv` usando il suffisso del filename (`_0` o `_1`)

Nota didattica: questo è un esempio di **data acquisition** + **data preparation**.

### 7.4 Perché serve uno script di preparazione dati

Per l’orale: il dato “in natura” spesso non è nel formato ideale. Lo script:

- standardizza struttura e naming
- crea split coerenti e stratificati
- rende il processo riproducibile (chiunque può rigenerare dataset e risultati)

### 7.5 Normalizzazione per-image (perché)

In [data.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/data.py) facciamo:

- x = (x - mean) / std calcolati sull’immagine stessa

Perché:

- le CXR possono avere esposizione/contrasto diverso
- normalizzare riduce la variabilità e aiuta la rete a concentrarsi su struttura/anatomia

### 7.3 Dataset sintetico (solo per test rapido)

- Script: [generate_synthetic.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/generate_synthetic.py)

Serve per verificare che:

- pipeline gira
- training/eval non crashano

Non è “real-world” da solo, ma è utile per sviluppo.

---

## 8) Struttura del repository (componentistica)

### 8.1 Panoramica cartelle

- `src/cv_medical/` = libreria principale (dataset, modello, training, metriche, utils)
- `scripts/` = entrypoint CLI (prepare data, train, evaluate, inference, report pdf)
- `data/` = dataset (sintetico e reale) nel formato atteso
- `runs/` = output degli esperimenti (config, checkpoint, metrics)

### 8.2 File importanti (mappa)

- Config: [config.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/config.py)
- Utilities: [utils.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/utils.py)
- Dataset/Preprocessing: [data.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/data.py)
- Modello: [mtl_unet.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/models/mtl_unet.py)
- Loss: [losses.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/losses.py)
- Metriche: [metrics.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/metrics.py)
- Post-process: [postprocess.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/postprocess.py)
- Train/Eval loop: [engine.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/engine.py)

### 8.3 Pattern di progettazione usati

- **Modular design**: separazione netta tra data/model/engine/metrics/postprocess
- **Single Responsibility**: ogni file fa “una cosa”
- **Script come CLI**: i file in `scripts/` sono entrypoint riproducibili
- **Reproducibility**: seed fissato, config salvata, checkpoint salvati

### 8.4 Che cosa c’è dentro `runs/` (e perché è importante)

Ogni run salva artefatti fondamentali:

- `config.json`: iperparametri e impostazioni (riproducibilità)
- `checkpoints/best.pt`: stato del modello migliore su validation (criterio: Dice su val)
- `checkpoints/last.pt`: ultimo stato a fine training
- `latest_metrics.json`: metriche train/val aggiornate
- `eval_test.json`: metriche su test (quando lanci `scripts/evaluate.py`)

Per l’esame: dimostra che sai fare experiment tracking e che i risultati sono replicabili.

### 8.5 Device: CPU vs GPU vs Apple Silicon (MPS)

In [utils.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/utils.py) selezioniamo automaticamente il device:

- CUDA se disponibile (GPU NVIDIA)
- altrimenti MPS se disponibile (Apple Silicon)
- altrimenti CPU

All’orale puoi dire: “il training è accelerabile su GPU/MPS, ma il progetto funziona anche su CPU”.

---

## 9) Comandi (cosa dire e cosa fare)

### 9.1 Setup ambiente

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 9.2 Preparare dataset reale

```bash
.venv/bin/python scripts/prepare_montgomery.py --out data/montgomery --max-samples 0 --seed 42
```

### 9.3 Training

```bash
.venv/bin/python scripts/train.py --data-root data/montgomery --run-dir runs/montgomery --epochs 10 --image-size 256 --batch-size 8
```

Output:

- `runs/montgomery/checkpoints/best.pt`
- `runs/montgomery/checkpoints/last.pt`
- `runs/montgomery/latest_metrics.json`

### 9.4 Evaluation

```bash
.venv/bin/python scripts/evaluate.py --data-root data/montgomery --split test --checkpoint runs/montgomery/checkpoints/best.pt --out runs/montgomery/eval_test.json
```

### 9.5 Generare il PDF tecnico

```bash
.venv/bin/python scripts/make_report.py --eval-json runs/montgomery/eval_test.json --out-pdf Technical_Analysis.pdf --project-title "Medical Image Analysis Tool (Segmentation + Classification)"
```

### 9.6 Inferenza su singola immagine

```bash
.venv/bin/python scripts/infer.py --image path/to/image.png --checkpoint runs/montgomery/checkpoints/best.pt --out-dir runs/montgomery/infer
```

---

## 10) Come spiegare il flusso end-to-end (script narrativo per orale)

Esempio di esposizione:

1. “Carico immagini e maschere. Applico denoise, resize e normalizzazione. In training aggiungo augmentation geometrica.”
2. “Uso una UNet multi-task: un encoder condiviso, decoder per segmentation e una head di classificazione sul bottleneck.”
3. “Ottimizzo una loss combinata: BCE per class + BCE + Dice per seg. Salvo best/last checkpoint.”
4. “Valuto segmentation con Dice/mIoU e classification con Accuracy/Precision/Recall/F1 + Confusion Matrix.”
5. “Applico post-processing alla maschera (morfologia + largest component) per ridurre falsi positivi e buchi.”
6. “Genero un PDF con metodologia, risultati, failure analysis ed ethical considerations.”

### 10.1 Come leggere e commentare i risultati (parte orale)

Quando presenti le metriche, dì cosa significa “buono”:

- Segmentation: Dice/mIoU più alti = maschera più sovrapposta al ground truth.
- Classification: guardi confusion matrix, non solo accuracy.

Se ti chiedono perché i risultati non sono perfetti, argomenta con cause realistiche:

- dataset piccolo o sbilanciato
- immagini con qualità diversa (rumore, contrasto)
- domain shift (ospedale/macchina/protocollo)
- label noise (label binaria non sempre corrisponde a un pattern visivo univoco)

E proponi miglioramenti (senza promettere miracoli):

- più epoche e tuning di learning rate
- augmentation anche fotometrica (contrasto/luminosità) oltre a geometrica
- scelta soglia su validation per ottimizzare un obiettivo (F1 o recall)
- fine-tuning su un sottoinsieme del dominio target

---

## 11) Comparativa con la comanda (requirements → dove stanno nel progetto)

### 11.1 Pipeline Requirements (obbligatorio)

1. **Data Acquisition & Preprocessing**

- Dataset reale: [prepare_montgomery.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/prepare_montgomery.py)
- Preprocessing/augmentation: [data.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/data.py)

2. **Feature Engineering/Representation**

- Feature learned (CNN/UNet encoder): [mtl_unet.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/models/mtl_unet.py)

3. **Core Logic: classification/detection/segmentation/generation**

- Qui: **segmentation + classification** (multi-task): [mtl_unet.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/models/mtl_unet.py)

4. **Post-processing**

- Morfologia + componenti connessi: [postprocess.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/postprocess.py)
- Usato in inferenza: [infer.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/infer.py)

### 11.2 Implementation Standards (obbligatorio)

- Python + librerie industry-standard:
  - OpenCV (cv2), PyTorch, scikit-learn
- Codice modulare/clean/well-commented:
  - moduli separati e docstring nei file principali

### 11.3 Performance Evaluation (obbligatorio)

- Segmentation: Dice, mIoU
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix  
  Implementazione: [engine.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/src/cv_medical/engine.py) + [evaluate.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/evaluate.py)

### 11.4 Deliverables (obbligatorio)

- Repo GitHub pubblico (gestito da te)
- Source code: `src/`, `scripts/`
- requirements: [requirements.txt](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/requirements.txt)
- README: [README.md](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/README.md)
- PDF tecnico ≤10 pagine: [Technical_Analysis.pdf](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/Technical_Analysis.pdf)
  - Generazione: [make_report.py](file:///Users/giuseppegabrieledileo/Desktop/My_Works/exam-assignments/S00004924_computer-vision-project/scripts/make_report.py)

### 11.5 “Well-documented” (come lo dimostri in pratica)

Oltre a README e PDF, puoi sottolineare:

- script CLI riproducibili (chiunque può rigenerare dataset e risultati)
- cartella `runs/` con config, checkpoint e metriche (auditabilità)
- separazione netta dei moduli (data/model/engine/metrics/postprocess) che rende il codice leggibile

---

## 12) Domande tipiche da orale (con risposta)

### “Perché hai usato Dice oltre a BCE per la segmentazione?”

Perché Dice misura direttamente la sovrapposizione ROI, è robusto quando il background domina (class imbalance).

### “Perché multi-task invece di due modelli separati?”

Encoder condiviso → feature riutilizzate, meno parametri, potenziale regolarizzazione e output interpretabile via maschera.

### “Perché post-processing con morfologia?”

Perché i modelli producono rumore: opening rimuove piccoli falsi positivi, closing riempie buchi, largest component elimina frammenti.

### “Che cos’è la confusion matrix e perché è utile?”

Mostra TP/TN/FP/FN. In ambito medico è cruciale controllare FN (casi persi) e FP (allarmi inutili).

### “Perché fai augmentation solo sul training?”

Perché l’augmentation serve a regolarizzare il modello, ma:

- su validation/test vuoi misurare performance su dati “come sono”
- applicarla al test altererebbe la valutazione e renderebbe i risultati meno comparabili

### “Come scegli il checkpoint migliore?”

Nel training loop si salva `best.pt` quando migliora la metrica su validation (Dice).  
Questo evita di usare il test per la selezione (data leakage).

### “Cos’è AdamW e perché weight decay?”

- Adam è un ottimizzatore adattivo (adatta l’update per ogni parametro).
- AdamW applica weight decay in modo più corretto rispetto ad Adam classico.
  Il weight decay (regolarizzazione) tende a ridurre overfitting.

### “Perché normalizzi l’immagine?”

Normalizzare stabilizza il training e riduce variabilità tra immagini (esposizione/contrasto).  
In radiologia è comune perché la stessa anatomia può apparire con intensità diverse.

### “Perché usare anche mIoU oltre a Dice?”

Sono metriche simili ma non identiche:

- Dice enfatizza la sovrapposizione (2\*intersezione / somma aree).
- IoU usa intersezione su unione (più severa in alcuni casi).
  Riportarle entrambe dà un quadro più completo.

---

## 13) Checklist finale per presentazione

- Mostra la struttura repo e spiega a cosa serve ogni cartella (`src/`, `scripts/`, `data/`, `runs/`).
- Fai vedere 1-2 esempi di inferenza (immagine, maschera overlay, probabilità).
- Mostra `eval_test.json` e le metriche nel PDF.
- Spiega: preprocessing, UNet, multi-task, loss, metriche, post-processing, etica/privacy.

---

## 14) Mini-glossario (termini che possono chiederti)

- ROI: Region Of Interest, la parte dell’immagine importante.
- Mask: immagine binaria che indica pixel ROI vs background.
- Backbone: parte del modello che estrae feature (encoder).
- Head: parte finale specifica del task (seg head / cls head).
- Feature map: mappa di attivazioni prodotta da una convoluzione.
- Logit: output non normalizzato; sigmoid → probabilità.
- Iperparametri: parametri non appresi (LR, batch size, epochs, threshold…).
- Checkpoint: file con i pesi del modello salvati.
- Regularization: tecniche per ridurre overfitting (augmentation, dropout, weight decay).
- Domain shift: dati diversi dal training (ospedale, macchina, protocollo) → performance cala.
- Data leakage: usare info del test durante scelta parametri → risultati falsati.

### 14.1 Terminologia della comanda (copertura completa, anche se non usata nel nostro progetto)

Questa lista serve per non farti trovare impreparato se il docente cita “termini random” dalle slide.

#### Pipeline (termini)

- Data acquisition: come “recupero” i dati (download dataset, organizzazione cartelle, parsing label).
- Preprocessing: trasformazioni prima del modello (resize, denoise, normalizzazione, ecc.).
- Noise reduction (riduzione rumore): filtri (es. median blur) per ridurre artefatti/puntini.
- Normalization: rendere intensità confrontabili (es. (x-mean)/std) per stabilizzare training.
- Augmentation: trasformazioni artificiali SOLO in training per aumentare variabilità.
- Feature engineering/representation: come rappresenti l’immagine prima del “ragionamento” del modello.
  - Handcrafted features: descrittori progettati a mano.
  - Learned features: feature apprese da una rete neurale (CNN/Transformer).
- Core logic: il task principale (classification, detection, segmentation, generation).
- Post-processing: “ripulire” l’output dopo il modello.

#### Handcrafted features (se te lo chiedono)

Nel nostro progetto NON le usiamo (usiamo learned features), ma le slide citano esempi:

- HOG (Histogram of Oriented Gradients): descrive la distribuzione degli orientamenti dei gradienti (utile per forme/bordi).
- SIFT (Scale-Invariant Feature Transform): keypoints + descrittori robusti a scala/rotazione (classico per matching).
  Nota: SIFT è “storico” e spesso non è incluso di default per questioni di licenza nelle versioni vecchie; oggi molte volte si usa ORB come alternativa.

#### Learned features: CNN e Transformer

- CNN: usa convoluzioni; ottima per immagini, efficiente.
- Transformer / ViT: usa self-attention su patch; potente ma spesso più “pesante”.

#### Task alternativi della comanda (definizioni rapide)

- Classification: predice una classe per immagine (nel nostro progetto: 0/1).
- Segmentation: predice una classe per pixel (nel nostro progetto: maschera polmoni).
- Object detection: predice bounding box + classe per oggetto (es. persone/auto).
- Video analysis: lavora su sequenze; include tracking, optical flow, ecc.
- Generation: genera immagini (GAN/diffusion, image-to-image, ecc.).

#### Post-processing: NMS e morfologia (perché nelle slide)

- NMS (Non-Maximum Suppression): tipico in detection per eliminare box duplicati e tenere i più “forti”.
- Morfologia (opening/closing): tipico in segmentation per togliere rumore e riempire buchi.

#### Metriche (comanda) — cosa sono e quando si usano

- Accuracy: (corretti / totale). Può ingannare con class imbalance.
- Precision: TP/(TP+FP). “Quanti positivi predetti sono veri”.
- Recall (sensibilità): TP/(TP+FN). “Quanti positivi veri riesco a trovare”.
- F1: media armonica tra precision e recall (bilancia FP e FN).
- Confusion Matrix: matrice con TN/FP/FN/TP.
- IoU (Intersection over Union): overlap tra maschere o box.
- mIoU (mean IoU): IoU media su immagini (e/o classi).
- Dice: overlap alternativo (molto usato in medical imaging).
- mAP (mean Average Precision): tipica della detection; misura qualità precision/recall su varie soglie.
- FID / Inception Score: tipiche di generation; misurano qualità/realismo (o distribuzione) delle immagini generate.

#### Deliverables / tooling

- requirements.txt: lista pacchetti Python per ricreare l’ambiente con pip (noi usiamo questo).
- environment.yml: alternativa con conda (vale uno dei due).
- Technical Analysis Document (PDF): documento tecnico ≤10 pagine con sezioni richieste (problem, methodology, results, failure, ethics).
- README “comprehensive”: come si installa, come si esegue, pipeline, risultati, link utili (dataset/Colab se usato).

### 14.2 Directory `colab/`: cos’è e a cosa serve

Abbiamo aggiunto una cartella `colab/` per soddisfare l’uso “opzionale” di Colab in modo pulito e riproducibile:

- Contiene un notebook: `colab/CV_Project_Train.ipynb`
- A cosa serve:
  - eseguire lo stesso progetto su Google Colab (GPU NVIDIA) senza cambiare il codice
  - salvare dataset + output su Google Drive (persistenza)
  - permettere al docente di riprodurre velocemente training/eval/report
- Cosa fa il notebook (in ordine):
  1. verifica GPU (`nvidia-smi`)
  2. monta Drive (`drive.mount`)
  3. definisce path su Drive (dove salvare data e runs)
  4. clona il repo in `/content/repo` (veloce) e installa dipendenze
  5. prepara il dataset Montgomery
  6. allena, valuta e genera il PDF

---

## 15) Approfondimenti utili per “riempire” l’orale (senza inventare)

### 15.1 Perché segmentation + classification ha senso clinico

- La segmentazione dei polmoni riduce rumore di background (spalle, bordi, marcatori) e può rendere la classificazione più robusta.
- La maschera è una “spiegazione visuale” semplice: aiuta a capire su quale regione il sistema sta lavorando.

### 15.2 Perché servono sia metriche quantitative sia qualitative

Oltre ai numeri (Dice/F1), in medical imaging spesso si mostrano esempi:

- overlay corretti (buona segmentazione)
- overlay con errori tipici (sottosegmentazione ai bordi, buchi, frammenti)
  Questo rafforza failure analysis e rende l’esposizione più convincente.

### 15.3 Calibrazione e soglie (discorso “maturo”)

Un modello può essere accurato ma non ben calibrato (probabilità troppo ottimiste/pessimiste).
In un progetto didattico puoi dire:

- scelgo la soglia su validation in base all’obiettivo (es. alta recall per non perdere casi)
- la soglia può essere diversa da 0.5

### 15.4 Etica e privacy (cosa dire in modo concreto)

- Non salvare mai dati sensibili in log o nei nomi dei file.
- Usare dataset pubblici e de-identificati.
- Non presentare il modello come “diagnosi”: è un supporto decisionale con limiti.

### 15.5 Colab / Hugging Face / Google Drive: cosa sono e quando servono

Questa parte è utile se ti chiedono “perché non avete usato Colab?” oppure “come avreste fatto su cloud?”.

**Google Colab**

- È un ambiente cloud di Google per eseguire notebook Jupyter dal browser.
- Il motivo principale per usarlo è avere **GPU/TPU** senza configurazione locale.
- Tipico flusso: cloni il repo → installi dipendenze → carichi dataset → lanci training/eval.

**Google Drive (con Colab)**

- Si usa spesso per:
  - leggere/salvare dataset (soprattutto se grande)
  - salvare output (`runs/`, checkpoint, PDF) in modo persistente
- In Colab puoi “montare” Drive e lavorare come fosse una cartella locale.

**Hugging Face (HF)**
HF è un ecosistema con più servizi:

- **Hub**: hosting di modelli e dataset (come un “GitHub del ML”).
- **Spaces**: hosting di demo web (es. Gradio/Streamlit) per mostrare inferenza.
  Si usa quando vuoi condividere checkpoint/dataset o fare una demo online.

**Nella comanda è obbligatorio?**

- No. Nelle slide è scritto “If Colab is used, include a link to the workbook”: quindi è richiesto **solo se scegli Colab**.
- Il progetto è valido anche se tutto gira in locale.

**Noi possiamo runnare tutto in locale?**

- Sì: setup venv, install dipendenze, script di dataset/training/eval/PDF/inferenza funzionano anche su CPU.
- La differenza principale è la **velocità**: senza GPU può essere più lento, ma la pipeline e i deliverable restano corretti.

**Se avessimo voluto usarli “nel caso” (schema pratico)**

- Colab:
  1. notebook Colab
  2. `git clone` del repo
  3. `pip install -r requirements.txt`
  4. (opzionale) mount Google Drive per dataset e `runs/`
  5. esecuzione script (`scripts/train.py`, `scripts/evaluate.py`, `scripts/make_report.py`)
  6. nel README: link al notebook (come richiesto “if Colab is used”)
- Hugging Face:
  - carichi il checkpoint `best.pt` sul Hub e/o crei uno Space per una demo di inferenza.

### 15.6 Il nostro run su Colab (cosa abbiamo fatto davvero)

Nel nostro caso abbiamo effettivamente eseguito il progetto su Colab per avere GPU e persistenza su Drive.

- **GPU usata**: Tesla T4 (Colab).  
  Nota: se dopo il training esegui `nvidia-smi` e vedi 0% di utilizzo è normale: significa solo che in quel momento non c’è nessun processo che sta usando la GPU.
- **Persistenza**: abbiamo montato Google Drive e salvato dataset + risultati in una cartella su Drive (così non si perdono a fine sessione).
  - Link cartella Drive (artefatti):  
    https://drive.google.com/drive/u/0/folders/1qXj2uZX2uR_wSkueBEqiPSA9VLz-62tM

Passaggi (in sintesi):

1. `drive.mount('/content/drive')` per montare Drive
2. `git clone` del repository in `/content/repo`
3. `pip install -r requirements.txt`
4. Dataset: `python scripts/prepare_montgomery.py --out <DRIVE>/data/montgomery ...`
5. Training: `python scripts/train.py --data-root <DRIVE>/data/montgomery --run-dir <DRIVE>/runs/montgomery ...`
6. Evaluation: `python scripts/evaluate.py ...` → genera `eval_test.json`
7. PDF: `python scripts/make_report.py ...` → genera `Technical_Analysis.pdf`

### 15.7 Come interpretare i risultati (spiegato semplice, con il nostro esempio)

Esempio reale (test set) che hai ottenuto su Colab:

```
loss_total=1.5523
dice=0.7403
miou=0.5976
cls_accuracy=0.6667
cls_precision=0.5
cls_recall=0.3333
cls_f1=0.4
cls_confusion_matrix=[[5,1],[2,1]]
```

#### 15.7.1 Segmentation: Dice e mIoU

- **Dice = 0.74**: in media la maschera predetta “si sovrappone” abbastanza bene a quella vera.  
  È una metrica molto usata in medical imaging; 1.0 sarebbe perfetto.
- **mIoU = 0.60**: è più severa di Dice, quindi è normale che sia più bassa.  
  Dire “Dice ~0.74 e IoU ~0.60” è un risultato coerente.

Messaggio da orale:

- “La segmentazione è la parte più solida: le maschere dei polmoni vengono ricostruite bene.”

#### 15.7.2 Classification: Accuracy, Precision, Recall, F1 e Confusion Matrix

Qui devi ragionare sulla confusion matrix (è la cosa più importante da spiegare):

- `[[5, 1], [2, 1]]` con convenzione (righe = vero, colonne = predetto):
  - **TN=5**: veri 0 predetti 0 (corretti)
  - **FP=1**: veri 0 predetti 1 (falso positivo)
  - **FN=2**: veri 1 predetti 0 (falso negativo)
  - **TP=1**: veri 1 predetti 1 (corretto)

Da qui derivano:

- **Accuracy = (TP+TN)/Totale = (1+5)/9 = 0.6667**
- **Precision = TP/(TP+FP) = 1/(1+1) = 0.5**
- **Recall = TP/(TP+FN) = 1/(1+2) = 0.3333**
- **F1** è basso perché la recall è bassa.

Come lo spieghi in modo semplice:

- “Il modello riconosce bene i negativi (TN alti), ma perde diversi positivi (FN=2).”
- “In ambito medico i falsi negativi sono importanti: significa non segnalare un caso positivo.”

Nota fondamentale per l’esame:

- il test set in questo esempio ha **solo 9 immagini**, quindi le metriche di classificazione sono “instabili”: basta 1 errore in più o in meno per cambiare molto precision/recall/F1.

#### 15.7.3 Cosa dire se ti chiedono “come miglioreresti la classificazione”

Risposte pratiche e credibili:

- scegliere una soglia diversa da 0.5 su validation (per aumentare recall e ridurre FN)
- gestire sbilanciamento classe (peso sulla loss o sampling bilanciato)
- aumentare dati/epoche, oppure validazione più robusta (k-fold) se il dataset è piccolo
- augment anche fotometrica (contrasto/luminosità) per CXR

### 15.8 Colab vs Locale: è la stessa cosa o cambia?

È **la stessa pipeline** e gli stessi script. Quello che cambia è l’ambiente di esecuzione.

Cose uguali:

- stessi file (`scripts/train.py`, `scripts/evaluate.py`, `scripts/make_report.py`, ecc.)
- stesso modello, preprocessing, loss, metriche, post-processing
- stessi output (checkpoints, json, pdf)

Cose che cambiano:

- **hardware**:
  - su Colab: CUDA (GPU NVIDIA, es. T4)
  - su Mac M1: MPS (Apple Metal) oppure CPU
- **performance**: Colab GPU spesso più veloce, soprattutto con più epoche o dataset completo
- **paths/persistenza**:
  - su Colab conviene salvare su Drive (`/content/drive/MyDrive/...`)
  - in locale salvi su disco (cartella `runs/` e `data/`)
- **riproducibilità numerica**: a parità di seed, piccoli scostamenti possono avvenire perché GPU/driver/kernel diversi portano a minime differenze floating-point

#### 15.8.1 Che cos’è CUDA (in parole semplici)

- **CUDA** è la tecnologia di NVIDIA che permette a programmi come PyTorch di eseguire i calcoli “pesanti” (soprattutto moltiplicazioni di matrici e convoluzioni delle CNN) direttamente sulla **GPU NVIDIA**.
- In pratica, quando in Colab abiliti “GPU”, PyTorch vede `cuda` e sposta tensori/modello su GPU: questo rende training/inferenza molto più veloci rispetto alla CPU.
- CUDA è “legata” alle GPU NVIDIA: su Mac non c’è CUDA.

#### 15.8.2 Che cos’è MPS (Apple Metal Performance Shaders)

- **MPS** è il backend di PyTorch per macOS che usa **Metal** (la tecnologia grafica Apple) per accelerare calcoli su Apple Silicon (M1/M2/M3…).
- In PyTorch lo vedi come device `mps`. Quando è disponibile, è spesso più veloce della CPU su molti modelli.
- In generale: CUDA (Colab + NVIDIA) è lo standard più comune nel deep learning; MPS è l’equivalente “Apple” per accelerare su Mac.

#### 15.8.3 Drive (Colab) vs Disco (locale): differenza pratica

- **Disco locale**: è il tuo storage sul computer (SSD/HDD). Se salvi `data/` e `runs/` lì, rimangono finché non li cancelli.
- **Google Drive**: è storage cloud collegato al tuo account Google. In Colab lo “monti” e lo usi come cartella (`/content/drive/MyDrive/...`).
- **Perché serve Drive in Colab**: la cartella `/content` di Colab è temporanea; se la sessione si resetta o scade, perdi file e risultati. Salvando su Drive hai **persistenza** (dataset, checkpoint, PDF).
- **Trade-off**: Drive è persistente ma può essere più lento del disco locale; per questo spesso si clona il repo in `/content` (veloce) e si salvano solo gli output importanti su Drive.

---

## 16) Copione pronto (2–3 minuti) + domande “trappola”

### 16.1 Copione pronto (2–3 minuti)

Puoi impararlo a memoria e adattarlo in base alle domande:

“Il mio progetto è una pipeline end-to-end di Computer Vision per immagini mediche, nello specifico chest X-ray.  
Ho implementato un sistema multi-task che svolge due compiti: segmentazione e classificazione.

Per la segmentazione, il modello produce una maschera binaria dei polmoni pixel-per-pixel; per la classificazione produce un’etichetta 0/1 associata a normal vs abnormal legata alla TB nel dataset scelto.

La pipeline parte dall’acquisizione e preparazione dati: scarico il dataset reale Montgomery County CXR e lo converto in un formato standard con train/val/test, immagini, maschere e labels.csv.  
In preprocessing carico le immagini in grayscale, riduco il rumore con median blur, ridimensiono a una dimensione fissa e normalizzo intensità per immagine. In training applico data augmentation geometrica per migliorare la generalizzazione.

Il core model è una UNet: encoder-decoder con skip connections. Ho un encoder condiviso e due head: una head di segmentazione e una head di classificazione che usa global average pooling sulle feature del bottleneck.  
Per l’ottimizzazione uso una loss combinata: BCEWithLogits per la classificazione e, per la segmentazione, BCEWithLogits + Soft Dice, perché Dice è robusta quando la ROI è piccola rispetto al background.

In valutazione misuro la segmentazione con Dice e mIoU, e la classificazione con Accuracy, Precision, Recall, F1 e confusion matrix.  
Infine applico post-processing alla maschera: threshold, opening/closing e largest connected component per rifinire l’output, come richiesto dalla comanda.  
Genero anche un Technical Analysis PDF con problem statement, metodologia, risultati, failure analysis ed ethical considerations, e rendo tutto riproducibile tramite script CLI e artefatti in runs/.”

### 16.2 Copione breve (30–45 secondi) “se mi interrompono”

“È un sistema multi-task su CXR: segmenta i polmoni e classifica 0/1.  
Ho una pipeline completa: preprocessing + UNet con due head + post-processing morfologico.  
Valuto con Dice/mIoU per la segmentazione e con Accuracy/Precision/Recall/F1 + confusion matrix per la classificazione.  
Tutto è riproducibile con script, checkpoint e un PDF tecnico.”

### 16.3 Domande “trappola” tipiche (con risposta strutturata)

#### D1) “Perché dici che è ‘real-world’ se il dataset è pubblico?”

Risposta:

- “Real-world” significa che il problema è realistico (screening support su CXR) e i dati non sono sintetici.
- Un dataset pubblico può essere real-world se rappresenta acquisizioni cliniche reali e label/annotazioni reali.
- Per dimostrarlo: descrivo origine del dataset, tipo di immagini, maschere, e limiti (domain shift).

#### D2) “Perché non basta fare solo classification senza segmentation?”

Risposta:

- Segmentation riduce background e può rendere le feature più stabili.
- Offre interpretabilità: la maschera mostra dove il sistema sta focalizzando.
- È una parte comune di pipeline medicali (pre-step per analisi successive).

#### D3) “Stai segmentando i polmoni: come lo colleghi a TB?”

Risposta:

- La segmentazione è un compito anatomico di base che delimita l’area di interesse principale.
- La classificazione cerca segnali compatibili con TB; lavorare dentro la regione polmonare aiuta.
- In più, multi-task con encoder condiviso regolarizza e migliora robustezza.

#### D4) “Perché hai scelto threshold 0.5?”

Risposta:

- 0.5 è un default naturale per sigmoid.
- In pratica la soglia si seleziona su validation in base all’obiettivo (es. massimizzare F1 o aumentare recall per ridurre FN).
- La soglia è un iperparametro: posso giustificare la scelta e mostrare come cambia la confusion matrix.

#### D5) “Dice e IoU sono entrambe necessarie?”

Risposta:

- Sono metriche correlate ma non identiche: IoU è spesso più severa.
- Riportarle entrambe dà un quadro più completo; in particolare, IoU penalizza di più sovra/sottosegmentazioni.

#### D6) “Che rischio c’è nel dire ‘accuracy 0.9’?”

Risposta:

- L’accuracy può essere ingannevole con class imbalance.
- In ambito medico guardo confusion matrix e F1/recall per controllare FN.

#### D7) “Hai fatto data leakage?”

Risposta:

- No: train aggiorna pesi, val seleziona checkpoint/iperparametri, test è usato solo per valutazione finale.
- Inoltre, augmentation è solo sul training, non su validation/test.

#### D8) “Perché hai usato AdamW?”

Risposta:

- È stabile e standard per reti profonde.
- Il weight decay funziona come regolarizzazione per ridurre overfitting.
- È coerente con la comanda (librerie standard, implementazione pulita e riproducibile).

#### D9) “Quali sono i failure cases più importanti?”

Risposta:

- Segmentazione: bordi polmonari difficili, basso contrasto, immagini croppate, artefatti.
- Classificazione: domain shift, label noise, class imbalance.
- Mitigazioni: augmentation, tuning soglia, fine-tuning, validazione cross-domain.

#### D10) “Cosa diresti sul tema etico/privacy?”

Risposta:

- Privacy: dati de-identificati, niente PII in log, accesso controllato.
- Safety: è decision support, non diagnostica; comunicare limiti e incertezza.
- Bias: prestazioni possono variare per device/protocollo; idealmente metriche stratificate.
