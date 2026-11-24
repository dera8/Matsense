# Esperimento: ablazione dell'operatore (§7.7.1)

Deciso il 19/10/2026, prima di qualunque run di questo esperimento.

**Domanda.** L'effetto di MatSense sulla guida viene dalla trasformazione dell'intensità
calibrata o dalla parte geometrica dell'operatore (perdita di punti e spostamento lungo il raggio)?

**Configurazioni.** `intensity_only`: fattori di MatSense sull'intensità, nessun punto tolto o
spostato. `geometry_only`: perdita e spostamento come MatSense, intensità invariata. Più
`matsense` (operatore completo) e `unmodified`.

**Disegno.** Solo pioggia (con asciutto la parte geometrica è nulla per costruzione).
Blocco = (scenario, seed): unmodified, matsense, intensity_only, geometry_only + seconda
unmodified come riferimento di rumore, in ordine casuale. Scenari 15, 05, 09; seed come la
campagna principale: 3 x 10 x 5 = 150 run. Stesso codice (hash 1965ab803c39), stessi scenari
congelati. Output in `runs/matsense_ablation_2026-10`.

**Analisi.** Distanza di Fréchet dalla run unmodified del blocco, Wilcoxon appaiato sui 10
blocchi: intensity_only vs rumore, geometry_only vs rumore, intensity_only vs matsense,
geometry_only vs matsense, per scenario. Correzione di Holm sulla famiglia di questi 12 test.
Script: `tools/analyze_ablation.py` (fuori dall'hash del codice).
