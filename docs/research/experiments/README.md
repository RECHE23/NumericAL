# Expériences de l'audit du 15 octobre 2026

Prototypes et mesures de [l'audit de v2026.10.14](../audit-2026-10-15.md). Ce code de recherche ne fait pas partie de
la bibliothèque : il n'est ni testé par la porte de qualité ni installé.

Machine de mesure : VM KVM, 4 vCPU Intel Xeon Emerald Rapids (famille 6, modèle 207 ; AVX-512, AMX-INT8/BF16,
L1d 48 Kio, L2 2 Mio par cœur), GCC 13.3, NumPy 2.5.4 avec OpenBLAS 0.3.34. Depuis la racine du dépôt :

```bash
CXX="g++ -std=c++20 -O3 -Iinclude -pthread"
E=docs/research/experiments
```

| Fichier | Mesure | Construction et lancement |
| --- | --- | --- |
| `peak.cpp` | débits crêtes d'un cœur : FMA FP64 AVX-512, AMX INT8 et BF16 | `$CXX $E/peak.cpp -o peak && ./peak` |
| `drv.cpp` | pilote de profilage : une opération répétée (`NT` fixe les fils) | `$CXX -g -fno-omit-frame-pointer $E/drv.cpp -o drv && NT=1 perf record -g ./drv solve 1024 20` |
| `np1.py`, `mm1.py` | les mêmes opérations sous NumPy ; produits réels et complexes contre OpenBLAS | `python $E/mm1.py run c 4` |
| `trsv.cpp` | substitution avant : ligne par ligne contre blocs de 8 lignes, bits comparés | `$CXX $E/trsv.cpp -o trsv && ./trsv` |
| `narrow-solve.patch`, `solvecheck.cpp` | la substitution par blocs dans `lu_decomposition` : bits, écart, temps de `solve` | voir ci-dessous |
| `gemvt.cpp` | BLAS 2.5 : `A z` et `Aᵀ y` en un passage contre deux | `$CXX $E/gemvt.cpp -o gemvt && ./gemvt` |
| `amxk.cpp`, `amxk2.cpp` | noyau AMX-INT8 brut : blocage, opérandes en L1 | `$CXX -march=sapphirerapids $E/amxk.cpp -o amxk && ./amxk` |
| `ozaki2.cpp` | DGEMM émulée par Ozaki II sur AMX-INT8 : temps par étape, erreur selon le nombre de modules | `$CXX -march=sapphirerapids $E/ozaki2.cpp -o ozaki2 && ./ozaki2 1024 0 5` |

La substitution par blocs se mesure sur une copie corrigée des en-têtes ; les variables d'environnement `NARROW_OLD` et
`NARROW_FWD_ONLY` choisissent, dans le même processus, l'ancien chemin ou la seule passe avant par blocs :

```bash
mkdir -p /tmp/patched && cp -r include /tmp/patched/ && patch -p1 -d /tmp/patched < $E/narrow-solve.patch
g++ -std=c++20 -O3 -I/tmp/patched/include -pthread $E/solvecheck.cpp -o solvecheck && ./solvecheck
```

Le coût des défauts de page se mesure sans toucher au code, en empêchant la glibc de rendre la mémoire au système :

```bash
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824:glibc.malloc.top_pad=67108864 \
  NT=1 ./drv zmatmul 128 2000
```

L'AMX exige, sous Linux, la permission `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` que les prototypes
demandent eux-mêmes. Sous Valgrind, l'AVX-512 n'est pas émulé : la bibliothèque y prend son chemin AVX2.
