# Esperimento: camera degradata (caso limite)

Deciso il 17/10/2026, prima di vedere qualunque risultato di questo esperimento.

**Domanda.** Con la camera che non contribuisce alla nube LiDAR, la scelta del modello
LiDAR (unmodified, uniform, MatSense, shuffled) cambia la guida più che con la camera intatta?
In particolare: MatSense si separa da uniform e da shuffled?

**Manipolazione.** LAV "dipinge" ogni punto LiDAR con le classi predette dalla sua
segmentazione della camera. `tools/run_one_camera.py` sostituisce l'uscita della
segmentazione con "sfondo ovunque": i canali dipinti valgono 0. La rete gira comunque
(stesso tempo di calcolo); il classificatore di frenata dalla camera e il resto di LAV
non cambiano. Nessun file di PCLA o della campagna viene modificato. È un caso limite,
non una condizione meteo (come nel manoscritto, Sez. 7.6.1).

**Disegno.** Identico a `config/campaign.json` (stessi seed, stesse 4 configurazioni +
unmodified ripetuta, nominal e rain, stesso codice), solo scenari 05 e 09:
2 x 2 x 10 x 5 = 200 run. Output in `runs/matsense_camzero_2026-10`.

**Analisi.** `python analyze.py --config experiments/camera_zero/campaign.json`, poi
confronto con la campagna principale sugli stessi scenari e condizioni.
Ci si aspetta che l'agente sia degradato (meno rotta completata): il confronto è tra
esecuzioni degradate, e va riportato come tale.
