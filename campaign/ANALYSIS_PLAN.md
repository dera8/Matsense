# Piano di analisi della campagna principale

Fissato il 21/10/2026, dopo l'esecuzione degli scenari 05, 09 e 15 e **prima** di eseguire
gli scenari 16 e 18.

## Famiglia principale (test di ipotesi)

Domanda: a parità di livello di intensità, la struttura per materiale cambia la guida?

- Confronti: **MatSense contro uniform** e **MatSense contro shuffled**.
- Per ogni scenario (15, 05, 09, 16, 18) e condizione (nominal, rain): 5 × 2 × 2 = **20 test**.
- Metrica: distanza di Fréchet dalla run unmodified dello stesso blocco.
- Test: Wilcoxon appaiato sui 10 blocchi; **una correzione di Holm su tutti i 20 test**.
- Con 10 coppie il p minimo è 0,002; 20 × 0,002 = 0,039 < 0,05.

## Controlli di manipolazione (riportati a parte)

Ogni realizzazione contro il rumore (seconda run unmodified dello stesso blocco), per scenario
e condizione: dicono se l'esperimento sposta il veicolo oltre la variabilità del sistema.
Holm sulla loro famiglia (3 × 5 × 2 = 30 test), riportata separatamente.

## Esiti categorici

Completamento della rotta e categoria di terminazione: descrittivi; eventuali test (Fisher)
sono esplorativi e dichiarati come tali.

## Nota di trasparenza

La regola di `analyze.py` (Holm su tutti i 10 test per scenario, tutte le famiglie insieme)
con tre o più scenari non può dare significatività con 10 seed. La famiglia principale è stata
fissata dopo aver visto 05, 09 e 15: va dichiarato nel testo. Per 16 e 18 è fissata prima.

Script: `tools/analyze_primary.py` (fuori dall'hash del codice), dopo `python analyze.py`.
