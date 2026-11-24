# Esperimento E5: secondo agente (TransFuser++)

Deciso il 10/11/2026, prima di qualunque run di questo esperimento.

**Domanda.** La sensibilità al modello LiDAR osservata con LAV dipende dall'agente?
TransFuser++ legge la LiDAR come conteggio di punti per cella (sopra 0,2 m dal suolo,
saturato a 5) e non legge l'intensità: con la camera intatta ci si aspetta che MatSense,
uniform e shuffled non si separino tra loro né dal rumore.

**Agente.** PCLA `tfv4_l6_0`: pesi `transfuserv4_pretrained/longest6/tfpp_all_0`
(Hugging Face MasoudJTehrani/PCLA, scaricati il 10/11/2026).

**Disegno.** Identico a `config/campaign.json`: 5 scenari x 2 condizioni x 10 seed x
5 esecuzioni (4 configurazioni + unmodified ripetuta) = 500 run, stesso profilo,
stesso operatore, stesso codice. Output in `runs/matsense_tfpp_2026-11`.

**Analisi.** Come la campagna principale: famiglia primaria di 20 test (MatSense vs
uniform e vs shuffled), Holm; controlli di manipolazione a parte; Â12 con bootstrap.
`python analyze.py --config experiments/second_agent/campaign.json` poi
`tools/analyze_primary.py` con lo stesso config.
