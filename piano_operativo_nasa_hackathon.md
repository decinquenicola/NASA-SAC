# Piano Operativo NASA Hackathon: Sistema "ARES-OP" (Autonomous Routing & EVA Science - Orienteering Planner)

## 1. Visione del Progetto e Vantaggio Competitivo
Con un team di **5 persone forti in Python e GIS**, il vostro punto di forza è la capacità di trattare **dati geospaziali reali (GeoTIFF, DEM, mappe spettrali e termiche)** e trasformarli in un modello matematico rigoroso.

Invece di perdere tempo su framework frontend complessi partendo da zero, useremo un'architettura **Python- centrica ad alto impatto visivo**:
* **GIS & Pre-processing:** `QGIS` + `GDAL` / `Rasterio` per allineare, ritagliare (clip) e ricampionare i layer NASA nella stessa griglia di coordinate.
* **Core Matematico (Ricerca Operativa):** `NumPy` + `SciPy` + `scikit-image` (per l'algoritmo $A^*$ su matrice raster) + `Google OR-Tools` (per risolvere il *Team Orienteering Problem* su grafo).
* **Frontend & Visualizzazione Interattiva (Web App):** `Streamlit` + `PyDeck` (basato su Deck.gl di Uber, permette di renderizzare terreni 3D, layer raster e percorsi vettoriali in WebGL scrivendo solo Python) + `Plotly` per il profilo altimetrico/consumo $\text{O}_2$ e la vista **Spacesuit Visor HUD**.

---

## 2. Divisione dei Ruoli (Team da 5 Persone)

Per massimizzare il parallelismo nelle 48 ore (13–15 Novembre), ogni membro ha un modulo indipendente con interfacce di input/output ben definite (file `.tif` in ingresso, `.geojson` e `.json` in uscita):

| Ruolo | Membri | Responsabilità Principali | Output Tecnico |
| :--- | :---: | :--- | :--- |
| **1. GIS & Planetary Data Lead** | **Persona 1** | Scarica i dati NASA del 13 novembre, li allinea nello stesso CRS (Sistema di Riferimento Marziano), ritaglia l'area di studio (es. $2\text{ km} \times 2\text{ km}$ su *Mount Sharp* - formazioni *boxwork*) ed esporta 4 GeoTIFF della stessa identica dimensione $(H \times W)$ pixel. | `dem.tif`, `thermal.tif`, `spectral.tif`, `optical_rgb.tif` |
| **2. Terrain & Risk Modeling (Python/GIS)** | **Persona 2** | Da `dem.tif` e `thermal.tif` calcola Pendenza ($\theta$), Rugosità, *Sky-View Factor* (schermatura radiazioni) e genera la **Mappa Costo/Rischio** usando la funzione di Tobler modificata per la gravità marziana ($0.38g$). Individua anche i *Safe Havens* (rifugi naturali contro tempeste solari). | Matrice costo `cost_surface.npy` + `risk_layer.tif` |
| **3. Operations Research & Routing Lead** | **Persona 3** | Estrae i Nodi (POI scientifici) dalla **Mappa Obiettivo**, calcola la matrice dei sentieri minimi tra tutte le coppie di nodi con $A^*$ su griglia e risolve il **Team Orienteering Problem (TOP)** con `OR-Tools` (Astronauta + Rover di supporto). | `optimal_eva_route.geojson` + telemetria JSON |
| **4. Interactive Web App & 3D Map Developer** | **Persona 4** | Sviluppa la dashboard in `Streamlit` + `PyDeck` / `Folium`: gestisce gli slider in tempo reale (Autonomia $\text{O}_2$, Livello Radiazioni, Pesi Scientifici), la mappa 3D/2D multi-layer e i grafici di telemetria con `Plotly`. | App Web interattiva (`app.py`) |
| **5. Spacesuit HUD, Science Storyteller & Pitch** | **Persona 5** | Recupera le foto reali ad altezza uomo (POV delle fotocamere *Mastcam* / *Navcam* di Curiosity/Perseverance), crea il componente grafico **Visor HUD** (visiera del casco in prima persona), scrive la documentazione scientifica e monta il video/presentazione finale. | Modulo HUD + Pitch Deck + Video Demo |

---

## 3. Modello Matematico e Geospaziale (Da citare nella presentazione)

### 3.1 Mappa Obiettivo e Generazione dei Nodi ($V$)
Siano $L_{\text{spec}}(x,y)$, $L_{\text{geo}}(x,y)$ e $L_{\text{therm\_anom}}(x,y)$ i layer normalizzati in $[0, 1]$ che rappresentano rispettivamente l'interesse mineralogico/spettrale (es. solfati/argille), la presenza di strutture geologiche target (es. reticoli *boxwork*) e le anomalie termiche.
Il valore scientifico del pixel $(x,y)$ è:
$$S(x,y) = w_1 L_{\text{spec}}(x,y) + w_2 L_{\text{geo}}(x,y) + w_3 L_{\text{therm\_anom}}(x,y)$$
Con $w_1 + w_2 + w_3 = 1$ (modificabili dall'astronauta tramite slider).
Applichiamo una **maschera di accessibilità** $M(x,y) \in \{0, 1\}$ che vale $0$ se la pendenza locale $\theta(x,y) > \theta_{\max}$ (es. $20^\circ$) o se l'inerzia termica $I(x,y) < I_{\min}$ (sabbia profonda non calpestabile). I massimi locali di $S(x,y) \cdot M(x,y)$ definiscono l'insieme dei nodi candidati $V = \{0, 1, \dots, N\}$, dove $0$ è la Base/Lander.

### 3.2 Mappa Rischio e Costo degli Archi ($c_{ij}$) con Funzione di Tobler Marziana
Per ogni pixel $(x,y)$, la pendenza $\tan(\theta) = \sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2}$ determina la velocità di camminata. Partendo dalla **Tobler's Hiking Function** terrestre e adattandola a una tuta pressurizzata EVA su Marte ($g_{\text{Mars}} \approx 3.72\text{ m/s}^2$), la velocità base $v(x,y)$ in $\text{m/s}$ è:
$$v(x,y) = v_{\text{EVA}} \cdot \exp\left(-3.5 \left| \tan\theta(x,y) + 0.05 \right|\right) \cdot \phi_{\text{terreno}}(x,y)$$
dove:
* $v_{\text{EVA}} \approx 0.8\text{ m/s}$ è la velocità nominale in piano con tuta pressurizzata.
* $\phi_{\text{terreno}}(x,y) \in [0.2, 1.0]$ è il coefficiente di penalità derivato dall'**inerzia termica** (sabbia fine rallenta la marcia) e dalla **rugosità** topografica.

Il costo di attraversamento di un singolo pixel di dimensione $\Delta s$ (in metri) combina **tempo metabolico/ossigeno** e **esposizione alle radiazioni** $R(x,y)$:
$$\text{Cost}(x,y) = \frac{\Delta s}{v(x,y)} \cdot \left(1 + \alpha_{\text{rad}} R(x,y)\right)$$
L'algoritmo $A^*$ (o *Minimum Cost Path* su griglia 8-connessa) trova il percorso ottimale $\pi_{ij}$ tra ogni coppia di waypoint $(i, j) \in V \times V$, restituendo il tempo/costo effettivo dell'arco $c_{ij}$ e la polilinea geometrica che aggira massi, crateri e pendenze proibitive.

### 3.3 Formulazione dell'Orienteering Problem (OP / TOP)
Dato un tempo/autonomia massima di missione $T_{\max}$ (es. $240\text{ minuti}$ di ossigeno utilizzabile) e un tempo di campionamento scientifico $t_i^{\text{sci}}$ necessario per analizzare il nodo $i$:
$$\max \sum_{i=1}^{N} S_i y_i$$
soggetto a:
$$\sum_{i=0}^{N} \sum_{j=0, j \neq i}^{N} \left(c_{ij} + t_j^{\text{sci}}\right) x_{ij} \le T_{\max}$$
$$\sum_{j=1}^{N} x_{0j} = \sum_{i=1}^{N} x_{i0} = 1$$
$$\sum_{i \neq k} x_{ik} = \sum_{j \neq k} x_{kj} = y_k \quad \forall k \in \{1, \dots, N\}$$
più i vincoli di eliminazione dei sottocicli (MTZ) e il **vincolo di Safe Haven**: da qualsiasi nodo visitato $k$ ($y_k=1$), la distanza temporale verso il rifugio anti-radiazioni più vicino deve essere inferiore a $T_{\text{shelter}}$ (es. $15\text{ minuti}$) in caso di evento protonico solare (SPE).

---

## 4. Roadmap Operativa: Cosa fare PRIMA del 13 Novembre (3 – 12 Novembre)

Non aspettate il 13 novembre per costruire il software. Arrivate all'inizio dell'hackathon con la piattaforma funzionante al 100% su un dataset di test.

### Giorni 1–3 (Setup & Dati di Prova)
1. **Ambiente Python condiviso (GitHub):** Create il repository con `rasterio`, `numpy`, `scipy`, `scikit-image`, `ortools`, `geopandas`, `shapely`, `streamlit`, `pydeck`, `plotly`.
2. **Download Dataset di Test su Mount Sharp (Gale Crater):**
   * Il testo della challenge cita esplicitamente le **boxwork formations su Mount Sharp studiate da Curiosity nel 2025**.
   * Andate su **NASA Mars Trek** (`https://trek.nasa.gov/mars/`) o sull'archivio **USGS Astrogeology / HiRISE DTM** e scaricate un ritaglio DEM + Immagine Ottica della regione di *Mount Sharp / Gediz Vallis / Boxwork ridge*.
3. **Esecuzione dello Script Base (`mars_eva_engine.py`):** Testate lo script allegato qui sotto per verificare che l'intera catena (Raster $\rightarrow$ Nodi $\rightarrow$ $A^*$ $\rightarrow$ OR-Tools $\rightarrow$ GeoJSON) giri in meno di 3 secondi.

### Giorni 4–7 (Sviluppo Interfaccia & Modalità HUD)
1. **Persona 4** costruisce la dashboard `Streamlit` che legge il file `optimal_eva_route.geojson` generato dallo script e lo mostra in 3D sopra il DEM.
2. **Persona 5** scarica 10 foto ad alta risoluzione dalla **NASA Mars Rover Photos API** (Curiosity Mastcam) e prepara l'interfaccia **Spacesuit Visor HUD** (un'immagine panoramica marziana con overlay vettoriale in stile visiera spaziale: bussola in alto, barra $\text{O}_2$, indicatore radiazioni in $\mu\text{Sv/h}$, e poligono in realtà aumentata che indica dove colpire con il martello geologico).

---

## 5. Cronoprogramma delle 48 Ore (13 – 15 Novembre)

### Giorno 1 — 13 Novembre (Sblocco Resources & Ingestione Dati)
* **Ore 09:00 – 11:00:** Analisi delle *Resources* ufficiali. Scelta definitiva della bounding box marziana (consigliato: *Mount Sharp Boxwork Region* o *Jezero Crater Delta*).
* **Ore 11:00 – 15:00 (Persona 1 & 2):** In QGIS, ritaglio e allineamento di almeno **4 layer** ufficiali (Topografico DEM, Termico THEMIS, Spettrale CRISM/chimico, Ottico HiRISE). Esportazione dei `.tif` allineati al pixel.
* **Ore 15:00 – 19:00 (Persona 2 & 3):** Calibrazione dei pesi sulla nuova area geografica. Esecuzione dell'estrattore di waypoint e verifica che i percorsi $A^*$ seguano canaloni realistici evitando pareti ripide.
* **Ore 19:00 – 22:00 (Tutti):** Primo test end-to-end con i dati ufficiali dell'hackathon caricati nella Web App.

### Giorno 2 — 14 Novembre (Feature "Wow" & Caso d'Uso Scientifico)
* **Mattina (Persona 3):** Aggiunta della modalità **Team Orienteering Problem (Astronauta + Rover)** o **Simulazione Emergenza Solare**: un bottone *"Trigger Solar Storm Alert"* nella UI che ricalcola istantaneamente il percorso forzando il rientro immediato al *Safe Haven* (cavità o parete rocciosa schermata) più vicino.
* **Pomeriggio (Persona 4 & 5):** Collegamento dei Waypoint sulla mappa 2D/3D con le foto POV reali della superficie: cliccando sul Waypoint #3 sulla mappa, si apre la vista **Visor HUD** di quel punto esatto.
* **Sera (Persona 5 + Tutti):** Scrittura del report scientifico (perché abbiamo scelto quei layer, come funziona la matematica dell'Orienteering Problem, come garantiamo la sicurezza dell'astronauta).

### Giorno 3 — 15 Novembre (Rifinitura, Video Demo & Consegna)
* **Mattina:** Freeze del codice. Registrazione del video demo (spesso decisivo per i giudici NASA: mostrare in 2 minuti la selezione dei layer, il calcolo del percorso ottimo al variare dell'ossigeno e il passaggio alla visuale casco HUD).
* **Pomeriggio:** Controllo link, repository pubblico, documentazione e sottomissione finale con almeno 2 ore di margine.