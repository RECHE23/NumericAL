# Dépasser BLAS/LAPACK en C++ pur, sans dépendance — revue de littérature

*Octobre 2026. Pistes classiques, atypiques et expérimentales pour qu'une bibliothèque header-only, générique et
sans dépendance (NumericAL) rattrape puis dépasse OpenBLAS, MKL et Accelerate.*

Point de départ : les écarts mesurés dans [BENCHMARKS.md](../../BENCHMARKS.md) (2026-10-12) :

- petits problèmes (n ≤ 256) : 1,5 à 4 fois plus lents ;
- un cœur x86-64 à environ 0,7 fois le débit d'OpenBLAS ;
- `eigh` 2,6 à 2,9 fois plus lent sur x86-64 dès n ≥ 512 ;
- `svd` et `lstsq` 1,3 à 2,3 fois plus lents ;
- les produits complexes passent par une tuile scalaire (aucun chemin AVX ni AMX).

**Méthode.** Recherche en ligne (articles, prépublications arXiv, actes SC, PPoPP et IPDPS, revues ACM TOMS, SIAM,
Nature). Les chiffres cités sont **ceux que rapportent les auteurs**, sur leur matériel : ce sont des ordres de
grandeur, à re-mesurer par `make bench-linalg` avant toute adoption. Les estimations propres à NumericAL sont
marquées *(est.)*.

---

## 0. En bref

Chaque piste porte une étiquette de compatibilité avec la discipline numérique de la bibliothèque
(README, « Numerical discipline ») :

- **B** : mêmes bits que le calcul actuel (même ordre de sommation) ;
- **D** : déterministe (mêmes bits quel que soit le nombre de fils), bits différents du calcul classique, stabilité
  inverse conservée ;
- **C** : stabilité conservée sous condition, avec un repli automatique ;
- **O** : borne d'erreur affaiblie (normwise au lieu de componentwise), donc proposée seulement sur demande (opt-in).

| # | Piste | Opérations visées | Gain attendu | Effort | Contrat |
| --: | --- | --- | --- | --- | :-: |
| 1 | `lstsq` par polyalgorithme : QR/LQ, estimateur de conditionnement, repli SVD (§3.4) | `lstsq` | 2–4× sur l'actuel *(est.)*, plus rapide que `gelsd` | faible | D |
| 2 | Produits complexes sur les noyaux réels, méthodes 1M/4M (§2.5) | tout `complex128` | plusieurs × *(est.)* : la tuile est scalaire aujourd'hui | faible–moyen | D |
| 3 | Blocage par jeu d'instructions selon le modèle analytique de BLIS (§2.1) | GEMM et BLAS3 sur x86 | vise l'écart 0,7× d'un cœur | faible | **B** |
| 4 | Fusion « BLAS 2.5 » et `symv`/`gemv` vectorisés (§5.2) | `svd`, `lstsq`, `eigh` | jusqu'à ½ du trafic mémoire de l'étape BLAS2 | moyen | D |
| 5 | Ordonnanceur : modèle de coût par forme, région parallèle persistante (§2.2) | n ≤ 256 | 1,5–2× aux petites tailles *(est.)* | moyen | **B** |
| 6 | Raffinement itératif en précision mixte, comme `dsgesv` (§4.1) | `solve`, résolution de Cholesky | 1,3–2× pour n ≥ 512 | faible | C |
| 7 | Réduction en deux étapes : bande puis tridiagonale (§5.1) | `eigh`, valeurs propres seules | grand sans vecteurs ; avec vecteurs, à mesurer | élevé | D |
| 8 | Algorithmes récursifs (ReLAPACK) ; Gauss-Jordan par blocs pour l'inverse (§5.6, §5.7) | LU, Cholesky, `inverse` | moyennes tailles | moyen | D |
| 9 | Strassen « rechargé », fusionné au packing (§3.1) | GEMM, mises à jour de traîne | 10–25 % pour n ≳ 1000–2000 | moyen | O |
| 10 | Arm SME documentée via ACLE (Apple M4 et suivants) (§2.4) | BLAS3 sur Apple | pérennité ; FP32 à 4× le FP64 | moyen | D |
| 11 | **Expérimental** : Ozaki-II INT8 sur Intel AMX, Ogita–Aishima, D&C accéléré par HSS, QDWH, FOSSILS, RBT | GEMM, `eigh`, `svd`, `lstsq`, LU | de 0 à plusieurs × | élevé | C/D |

---

## 1. Cadre : cinq leviers

1. **Faire les mêmes flops plus vite** : noyaux, blocage, ordonnancement (§2).
2. **Faire moins de flops** : algorithmes rapides, polyalgorithmes (§3).
3. **Faire des flops moins chers** : précision mixte, émulation sur unités entières ou bf16 (§4).
4. **Mieux placer les flops** : convertir le BLAS2, limité par la mémoire, en BLAS3, limité par le calcul (§5).
5. **Changer de problème** : randomisation et sketching (§6).

Les bibliothèques de référence excellent sur le levier 1 pour les grandes tailles. Une bibliothèque header-only peut
gagner sur les leviers 2 à 5, que LAPACK n'exploite pas ou peu (NumPy appelle `dgesv`, `dsyevd` et `gelsd`, jamais
`dsgesv` ni un algorithme randomisé), ainsi que sur les petites tailles, où elle connaît les formes à la compilation.

---

## 2. Le noyau GEMM

### 2.1 Grands produits sur x86 : l'écart de 0,7× sur un cœur

- **Modèle analytique.** [Low, Igual, Smith, Quintana-Ortí, *Analytical modeling is enough for high-performance BLIS*,
  ACM TOMS 43(2), 2016](https://www.cs.utexas.edu/~flame/pubs/TOMS-BLIS-Analytical.pdf) dérive `kc`, `mc` et `nc` de la
  géométrie des caches. Le micro-panneau de B (`kc × nr`) doit occuper environ la moitié du L1, le bloc de A
  (`mc × kc`) une fraction du L2, et le panneau de B (`kc × nc`) le L3. Les mainteneurs de BLIS recommandent de
  confirmer le résultat par un balayage empirique.
- **Diagnostic probable** *(est., à vérifier)*. Le chemin AVX-512 a `nr = 16` et `kc = 256` (`gemm_kc`, commun aux
  chemins AVX2 et AVX-512). Le micro-panneau de B fait donc 256 × 16 × 8 o = 32 Kio, **tout le L1d** de l'i9-10900X,
  et le micro-panneau de A (24 Kio) qui défile l'en chasse. De plus, une tâche de `block_rows = 96` lignes ne
  réutilise chaque micro-panneau de B que 96/12 = 8 fois avant de passer au suivant. Le modèle suggère `kc ≈ 128`
  pour AVX-512, ou une tuile plus haute et moins large, et des tâches plus hautes. Ce réglage est **neutre au bit** :
  chaque élément de C reste sommé dans l'ordre croissant de la profondeur, et l'aller-retour en mémoire entre deux
  panneaux est exact.
- **Classiques à ne pas oublier.** [Goto et van de Geijn, *Anatomy of high-performance matrix multiplication*, ACM TOMS
  34(3), 2008](https://doi.org/10.1145/1356052.1356053) montre que les défauts de TLB comptent autant que ceux du cache.
  Trois remèdes en découlent :
  - des tampons de packing alloués sur de grandes pages (`madvise(MADV_HUGEPAGE)` sous Linux, sans dépendance) ;
  - des tampons alignés sans aliasing à 4 Kio ;
  - le préchargement (`prefetcht0`) du prochain micro-panneau et de la tuile de C, comme dans les noyaux d'OpenBLAS
    et de BLIS.
- **Preuve d'existence.** [Salykova, *Beating NumPy's matrix multiplication in 150 lines of C* (2024)](https://salykova.github.io/matmul-cpu)
  suit la structure de BLIS, en C pur plus OpenMP, et dépasse OpenBLAS (> 1 TFLOPS) sur un Ryzen 7700 8 cœurs, sans
  AVX-512. L'auteur prévient que ce résultat demande un réglage par machine.

### 2.2 Petits produits (n ≤ 256) : sans packing et sans réveil de fils

- **Sans packing.** Le chemin *sup* de BLIS (*skinny/unpacked*) lit les opérandes en place
  ([Van Zee, BLIS Retreat 2019](https://cs.utexas.edu/~flame/BLISRetreat2019/slides/Field.pdf)).
  [GEMMFIP (2023)](https://arxiv.org/abs/2302.08417) fusionne le packing au premier passage de calcul : un seul code
  pour toutes les tailles.
- **Format panneau-majeur.** [BLASFEO (Frison et al., ACM TOMS 44(4), 2018)](https://arxiv.org/abs/1704.02457)
  stocke les matrices directement dans le format que lit le micro-noyau. Le packing disparaît et BLASFEO dépasse les
  BLAS optimisées pour les matrices qui tiennent en cache. Pour NumericAL, il suffirait de garder ce format *à
  l'intérieur* d'une décomposition, d'une mise à jour de traîne à la suivante, sans toucher l'API.
- **Formes irrégulières.** [LibShalom (Yang et al., SC21)](https://eprints.whiterose.ac.uk/177559/) atteint 1,05 à 3×
  OpenBLAS, ARMPL, BLIS et LIBXSMM sur ARMv8. AutoGEMM (SC24) annonce jusqu'à 2× sur LIBXSMM et LibShalom.
- **Spécialisation à la compilation.** [Fastor](https://github.com/romeric/Fastor), header-only, déroule entièrement
  les produits de tailles fixes. Une extension naturelle serait un `fixed_matrix<T, M, N>` dont les produits sont
  générés par templates.
- **Fils.** Le README le note : le seuil de parallélisme est un seul poids pour toutes les formes.
  - **Modèle de coût par forme** : choisir `p` qui minimise `t(p) = W / (p · R) + c₀ + c₁ · log₂ p`, avec `R`, `c₀`
    et `c₁` étalonnés au premier usage, à la manière du planificateur de FFTW
    ([Frigo et Johnson, Proc. IEEE 93(2), 2005](https://doi.org/10.1109/JPROC.2004.840301)). ISAAC
    ([Tillet et Cox, SC17](https://arxiv.org/abs/1802.05371)) apprend ce choix selon la forme de l'entrée.
  - **Région parallèle persistante** : une décomposition enchaîne des dizaines de noyaux. Une portée
    `parallel_region` qui garde l'équipe active, comme une région `omp parallel` qui englobe plusieurs boucles,
    évite un réveil par noyau.
  - **Barrière** : un drapeau local par fil, à renversement de sens ou en arbre
    ([Mellor-Crummey et Scott, ACM TOCS 9(1), 1991](https://www.cs.rochester.edu/u/scott/papers/1991_TOCS_synch.pdf)),
    plutôt qu'une `condition_variable` après la phase d'attente active.

### 2.3 Générer plutôt qu'écrire

- **Exo et Exo 2** ([Ikarashi et al., PLDI 2022](https://par.nsf.gov/servlets/purl/10342192) ; Exo 2, ASPLOS 2025)
  ordonnancent des noyaux depuis une bibliothèque utilisateur. Ses auteurs annoncent plus de 80 noyaux au niveau de
  MKL, OpenBLAS, BLIS et Halide. Usage sans dépendance : **générer hors ligne** les tuiles (AVX2, AVX-512, NEON, SME,
  RVV, bords partiels) et versionner le C produit dans les en-têtes.
- **JIT header-only.** [Xbyak](https://github.com/herumi/xbyak), un assembleur JIT x86 header-only (le moteur de
  oneDNN), montre que c'est faisable. Un émetteur minimal (load, broadcast, FMA, store) tient en quelques centaines de
  lignes. Le gain : spécialiser sur `m`, `n`, `k`, les pas et les bords réels, comme
  [LIBXSMM (Heinecke et al., SC16)](https://github.com/libxsmm/libxsmm), qui dépasse MKL aux petites tailles.
  Contraintes : W^X (`mmap` RW puis `mprotect` RX ; `MAP_JIT` et `pthread_jit_write_protect_np` sur macOS arm64), et
  certains environnements interdisent la mémoire exécutable. Le chemin compilé d'avance doit donc rester le repli.
- **Recherche par LLM ou évolutionnaire.** [QiMeng-GEMM (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34461)
  annonce 211 % d'OpenBLAS sur RISC-V (chiffre auto-déclaré). [AdaExplore (2026)](https://arxiv.org/abs/2604.16625)
  n'atteint que 0,42× cuBLAS sur GEMM : le GEMM reste difficile. Ces outils servent au mieux d'assistants pour
  explorer des variantes de tuiles, sous l'oracle `bench-linalg` + `fuzz-linalg`.

### 2.4 Unités matricielles documentées

- **Arm SME/SME2** (Apple M4 et suivants, futurs cœurs Arm) est le successeur *documenté* de l'AMX, que NumericAL
  utilise aujourd'hui sans documentation et seulement sur M1 à M4. Les versions récentes de Clang et de GCC
  l'exposent par `arm_sme.h`, `__arm_streaming` et `__arm_new("za")`.
  [*Hello SME!* (Remke et Breuer, SC24-W)](https://arxiv.org/abs/2409.18779) mesure plus de 2,3 TFLOPS FP32 sur M4, et
  des noyaux qui dépassent Accelerate en FP32. Le **FP64 tourne au quart du FP32** (comme l'AMX) ; l'INT8 et le BF16
  restent comparativement lents sur M4.
- **Intel AMX** (Xeon Sapphire Rapids et suivants) ne traite que BF16, INT8 et FP16, sans FP64 : il ne sert NumericAL
  que par l'émulation (§4.4).
- **`std::simd` (C++26).** [P1928](https://isocpp.org/files/papers/P1928R15.pdf), adoptée pour C++26 en novembre
  2024, remplacerait les extensions vectorielles de GCC et Clang dans le chemin portable. Elle apporterait aussi des
  vecteurs à MSVC, qui retombe aujourd'hui sur des tableaux. GCC fournit déjà `std::experimental::simd` depuis GCC 11.
  Les implémentations restent jeunes et certains critiquent leurs performances ; c'est une option pour la portabilité
  plus que pour la vitesse brute.

### 2.5 Nombres complexes : réutiliser les noyaux réels

La tuile complexe est un tableau scalaire : `complex128` ne profite ni d'AVX ni de l'AMX. Trois méthodes la ramènent
aux noyaux réels :

| Méthode | Idée | Flops | Stabilité | Référence |
| --- | --- | --- | --- | --- |
| **4M** | quatre produits réels (Re·Re, Im·Im, Re·Im, Im·Re) | égaux | celle du produit classique | [Van Zee et Smith, ACM TOMS 44(1), 2017](https://www.cs.utexas.edu/users/flame/pubs/blis5_toms_rev2.pdf) |
| **1M** | packing spécial, puis **un** appel au noyau réel | égaux | celle du produit classique | Van Zee, *The 1m method*, SIAM J. Sci. Comput. 42(5), 2020 ([pdf](https://www.cs.utexas.edu/users/flame/pubs/blis6_sisc_rev3.pdf)) |
| **3M** | `(a+b)(c+d) − ac − bd` | −25 % de multiplications | normwise seulement, sur la partie imaginaire | Higham, SIMAX 13(3), 1992 |

**Recommandation** : 1M par défaut (le même noyau réel sur tous les chemins, AMX compris), 3M en opt-in.

---

## 3. Faire moins de flops

### 3.1 Strassen « rechargé »

- [Huang, Smith, Henry, van de Geijn, *Strassen's algorithm reloaded*, SC16 (rapport arXiv)](https://arxiv.org/abs/1605.01078)
  fusionne les additions de Strassen au packing et au micro-noyau. Il ne demande aucun espace de travail
  supplémentaire et devient **rentable dès les tailles moyennes**, contrairement à l'idée reçue.
  [Huang et al., IPDPS 2017 (diapositives)](https://www.cs.utexas.edu/~flame/BLISRetreat2017/slides/Jianyu_BLIS_Retreat_2017.pdf)
  en génère des familles entières, à un ou plusieurs niveaux.
- [Benson et Ballard, PPoPP 2015](https://arxiv.org/abs/1409.2908) génère plus de 20 algorithmes rapides et dépasse
  MKL de plus de 15 % en parallèle ; le meilleur choix dépend de la taille *et* de la forme.
- [AlphaTensor (Fawzi et al., Nature 610, 2022)](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/)
  trouve des schémas 10 à 20 % plus rapides que l'algorithme usuel pour les grands produits, sur V100 et TPU.
- **Stabilité.** [Ballard et al., SIMAX 37(4), 2016](https://arxiv.org/abs/1507.00687) montre que la perte est modeste
  en pratique et que la **mise à l'échelle diagonale** des entrées la réduit. Le résultat reste déterministe mais
  perd l'égalité au bit avec la boucle i-k-j, d'où le contrat **O** : un `gemm_algorithm::strassen` à côté des
  `eigen_method` et `svd_method` existants.

### 3.2 Nouveaux schémas : IA et graphes de flips

- **4×4 en 48 multiplications.** AlphaEvolve (Google DeepMind, 2025) les trouve sur ℂ.
  [Dumas, Pernet et Sedoglavic (2025)](https://arxiv.org/abs/2506.13242) en donnent une version **à coefficients
  rationnels**, avec une variante en base alternative.
- **Graphes de flips.** [Kauers et Moosbauer (2022)](https://arxiv.org/abs/2212.01175) les introduisent ;
  [Moosbauer et Poole (2025)](https://arxiv.org/abs/2502.04514) multiplient 5×5 en 93 et 6×6 en 153 multiplications,
  et [Kauers et Wood (2025)](https://arxiv.org/html/2510.19787v1) améliorent une trentaine de formats rectangulaires.
- **Base alternative.** Karstadt et Schwartz (*Matrix multiplication, a little faster*, SPAA 2017) ramènent les
  additions de Strassen-Winograd de 15 à 12 et le coefficient dominant de 6 à 5.
- **Lecture réaliste.** Avec 48 multiplications contre 49 pour Strassen sur deux niveaux, le gain se perd dans les
  additions pour n ≤ 10⁴. C'est un terrain d'expérimentation, utile surtout si un générateur de code (§2.3) rend
  l'essai peu coûteux.

### 3.3 Produits structurés : AAᵀ et SYRK

- [Dumas, Pernet et Sedoglavic, ISSAC 2020](https://arxiv.org/abs/2001.04109) multiplient une matrice par sa
  transposée plus vite que par réduction au produit général. RXTX (Rybin, Zhang et Luo, 2025, trouvé par recherche
  automatique) retire encore environ 5 % des opérations.
- Cibles dans NumericAL : la mise à jour de traîne de Cholesky, CholeskyQR (§6.2), les équations normales.
  Contrat **O**, comme Strassen.

### 3.4 Polyalgorithmes : choisir l'algorithme selon le problème

- **`lstsq`** : NumPy appelle toujours `gelsd`, une SVD bidiagonale. NumericAL fait pareil et forme en plus `U` et
  `V`. Or, pour un `A` de rang plein, la solution des moindres carrés est **unique** : une QR de Householder (ou une
  LQ si m < n) donne la même solution, à l'arrondi près, pour environ la moitié des flops, et en BLAS3. La
  sémantique de NumPy (le seuil `rcond`) se conserve ainsi :
  1. QR non pivotée ;
  2. estimation du conditionnement de `R` (estimateur en norme 1 de Hager et Higham, comme `dtrcon`) ;
  3. si `κ_est` est sûrement sous `1 / (marge · rcond)`, résolution triangulaire, sinon **repli** sur la SVD
     actuelle.

  Avec le `rcond` par défaut (`ε · max(m, n)`), presque toutes les matrices passent le test. C'est le gain le plus sûr
  du document. Variante plus robuste : la QR à pivotage randomisé HQRRP (§6.3).
- **Détection de structure**, comme le polyalgorithme de `mldivide` dans MATLAB : une matrice triangulaire,
  diagonale, à bande, ou symétrique à diagonale positive (tenter Cholesky d'abord) se reconnaît en O(n²). C'est
  légitime à condition de le documenter ; sinon c'est un artifice de banc d'essai.

---

## 4. Faire des flops moins chers : précision mixte et émulation

Synthèses : [Higham et Mary, *Mixed precision algorithms in numerical linear algebra*, Acta Numerica 31,
2022](https://doi.org/10.1017/S0962492922000022), et le [survol de 2024](https://arxiv.org/abs/2412.19322).

### 4.1 Raffinement itératif : `solve` et résolution de Cholesky

- Le principe : factoriser en FP32 (deux fois plus de voies SIMD, moitié moins de trafic), puis corriger avec des
  résidus en FP64. LAPACK le fait déjà dans **`dsgesv`/`dsposv`** (Buttari, Dongarra, Langou et al., IJHPCA 21(4),
  2007), mais NumPy appelle `dgesv`.
- GMRES-IR ([Carson et Higham, SISC 40(2), 2018](https://doi.org/10.1137/17M1140819)) étend le domaine aux matrices
  mal conditionnées. Il existe aussi une version pour les moindres carrés (Carson, Higham et Pranesh, SISC 42(6),
  2020).
- Gains rapportés : 1,33 à 1,6× lorsque la factorisation domine, jusqu'à 2,2× sur certaines matrices, et des
  ralentissements (0,6×) pour les petites tailles ou les cas mal conditionnés
  ([synthèse](https://arxiv.org/abs/2412.19322), [diapositives de Th. Mary](https://perso.lip6.fr/Theo.Mary/doc/OxfordGreen2025.pdf)).
- Pour NumericAL : le contrat **C**. Le critère d'arrêt est l'erreur inverse componentwise d'Oettli–Prager ; après
  K itérations sans succès, on refactorise en FP64, comme `dsgesv`. À n'activer qu'au-delà d'un seuil de taille
  (n ≳ 512). `inverse` n'en profite pas : raffiner n colonnes coûte autant que l'inversion.

### 4.2 Raffinement d'Ogita–Aishima : `eigh` et `svd` en produits de matrices

- [Ogita et Aishima](https://www.keisu.t.u-tokyo.ac.jp/data/2016/METR16-11.pdf) partent d'une décomposition propre de
  faible précision et la raffinent par **quatre produits de matrices par itération**, avec une convergence
  quadratique (JJIAM 35, 2018 ; le cas des valeurs groupées dans JJIAM 36, 2019). La même idée s'applique à la SVD
  (J. Comput. Appl. Math. 369, 2020).
- **2026** : NVIDIA raffine une SVD de qualité FP32 en qualité FP64, jusqu'à 5,2× plus vite que la SVD FP64 de
  cuSOLVER ([NVIDIA Research](https://research.nvidia.com/publication/2026-06_mixed-precision-svd-gpus-ogita-aishima-iterative-refinement)).
  [Terao, Ozaki, Imamura et Ogita (2026)](https://arxiv.org/html/2602.23778v1) ne raffinent qu'un sous-ensemble de k
  vecteurs propres.
- **Lecture pour CPU** *(est.)* : le FP32 n'y est que 2× plus rapide (contre plus de 10× sur GPU). Sur x86, un `eigh`
  FP32 suivi de 1 à 2 itérations (≈ 7 n³ flops chacune, en GEMM) ne battrait probablement pas `dsyevd`. **Sur Apple
  M4 et ses successeurs**, où le FP32 de SME vaut 4 fois son FP64, la piste devient sérieuse. Limite connue :
  l'algorithme exige une approximation initiale assez précise quand des valeurs propres sont presque multiples.

### 4.3 Jacobi et MRRR en précision mixte

- [Higham, Tisseur, Webb et Zhou (2025)](https://arxiv.org/abs/2501.03742) calculent un préconditionneur (des vecteurs
  propres approchés) en basse précision, puis font Jacobi en précision de travail. Il suffit alors de quelques
  balayages, et les valeurs propres gagnent en **précision relative**. Les produits de matrices dominent le coût, ce
  qui convient à `eigen_method::jacobi` et `svd_method::jacobi`
  ([variante SVD](https://arxiv.org/html/2209.04626v2)).
- [Petschow, Quintana-Ortí et Bientinesi, SISC 36(2), 2014](https://arxiv.org/abs/1304.1864) calculent l'intérieur de
  MRRR en précision étendue (`long double` ou double-double, sans dépendance). MRRR (Dhillon, Parlett et Vömel, ACM
  TOMS 32(4), 2006) calcule les vecteurs d'une tridiagonale en O(n²). Avec cette précision mixte, il devient **au
  moins aussi précis que D&C et QR**, souvent plus rapide et plus parallèle. C'est une alternative crédible au
  divide-and-conquer actuel.

### 4.4 Émulation d'Ozaki : le FP64 sur unités entières ou bf16 (expérimental)

- **Ozaki-I** ([Ozaki, Ogita, Oishi et Rump, Numer. Algorithms 59, 2012](https://doi.org/10.1007/s11075-011-9478-1))
  découpe chaque opérande en tranches dont les produits sont exacts.
  **Ozaki-II** ([Ozaki, Uchino et Imamura, 2025](https://arxiv.org/abs/2504.08009)) passe par des résidus modulaires
  et le **théorème des restes chinois** : un produit entier exact par module, bien moins de produits au total, et une
  précision réglée par le nombre de modules.
- **Résultats.** Sur GPU INT8, l'émulation dépasse le DGEMM natif pour les grandes tailles
  ([Uchino et al., SC'25](https://arxiv.org/abs/2508.03984)) ; elle s'étend aux complexes, avec 4,4 à 6,5× sur ZGEMM
  ([déc. 2025](https://www.alphaxiv.org/abs/2512.08321)). La version à précision garantie de NVIDIA (ADP) estime à
  l'exécution le nombre de tranches nécessaires et **se replie sur le FP64** au besoin, pour moins de 10 % de surcoût
  ([arXiv 2511.13778](https://arxiv.org/abs/2511.13778)). Un correctif de 2026 répare une mise à l'échelle non
  invariante en mode rapide.
- **Sur CPU, Intel AMX** : [Cui et Liu (sept. 2026)](https://arxiv.org/abs/2609.04663) émulent FP32 et FP64 par
  produits de composantes BF16 :
  - FP32 : 1,14 à 2,56× la SGEMM de oneMKL ;
  - FP64 à 6 produits : jusqu'à 1,72× la DGEMM, mais avec une **précision réduite** et une plage d'exposants limitée
    à celle du BF16 ;
  - FP64 à 15 ou 21 produits : **pas plus rapide** que la DGEMM.
- **Piste non publiée** (à notre connaissance) : **Ozaki-II en INT8 sur AMX**. Intel annonce 2048 opérations INT8 par
  cycle et par cœur, contre 32 flops FP64 pour AVX-512 avec deux unités FMA, soit un rapport brut de 64. Il faut de
  l'ordre de 12 à 20 produits entiers pour une précision de niveau DGEMM (ordre de grandeur rapporté sur GPU, à
  confirmer), plus des conversions en O(n²). Plafond théorique : ≈ 3 à 5× pour les grandes tailles *(est.)*.
- **Atout inattendu pour NumericAL** : les accumulations entières sont **exactes**, donc le résultat ne dépend ni de
  l'ordre de sommation, ni du nombre de fils, **ni du noyau**. Un mode « reproductible inter-noyaux », même lent (un
  chemin entier portable), donnerait les mêmes bits sur portable, AVX2, AVX-512 et AMX. Le README reconnaît
  aujourd'hui des différences dans les derniers bits d'un noyau à l'autre.
- **Sur Apple**, l'INT8 et le BF16 de SME sont relativement lents (M4) : la piste n'y est pas rentable. Le levier y
  est le FP32 (§4.1 et §4.2).

---

## 5. Restructurer les décompositions : du BLAS2 au BLAS3

### 5.1 `eigh` : la réduction en deux étapes

- **Étape 1** : la matrice pleine devient une matrice à bande, entièrement en BLAS3. **Étape 2** : un *bulge chasing*
  ramène la bande à une tridiagonale, avec un accès mémoire local. Références : SBR de Bischof, Lang et Sun (ACM TOMS
  26(4), 2000) ; [Luszczek, Ltaief et Dongarra, IPDPS 2011](https://icl.utk.edu/~luszczek/pubs/ipdps2011trd.pdf)
  (jusqu'à 10× LAPACK multifil et 2,5× MKL) ; [Haidar, Ltaief et Dongarra, SC11](https://library.eecs.utk.edu/files/ut-cs-11-677.pdf) ;
  LAPACK `?sytrd_2stage` depuis la version 3.7.
- **Limite** : avec vecteurs propres, la rétro-transformation à travers l'étape 2 ajoute environ 2 n³ flops. Les
  pilotes `?syevd_2stage` de LAPACK de référence ne calculent d'ailleurs **que les valeurs propres**. ELPA2
  (Auckenthaler et al., Parallel Computing 37, 2011) fait la rétro-transformation efficacement et l'emporte sur de
  nombreux cœurs. Pour NumericAL, la piste est sûre pour un futur `eigvalsh`, et à mesurer pour `eigh` sur 8 à 10
  cœurs.
- **Remède immédiat**, déjà identifié dans BENCHMARKS.md : un chemin AVX2 ou AVX-512 pour le `symv` de la
  réduction. Il suffit d'étendre le dispatch existant ; GCC et Clang offrent aussi `target_clones` sur ELF.

### 5.2 Fusion « BLAS 2.5 » : bidiagonalisation et tridiagonalisation

- [Howell, Demmel, Fulton, Hammarling et Marmol, ACM TOMS 34(3), 2008](https://eprints.maths.manchester.ac.uk/210/)
  réordonnent la bidiagonalisation de Householder pour que ses **deux produits matrice-vecteur partagent un seul
  passage** sur la matrice de traîne. Le trafic de la mémoire vers le cache est environ **divisé par deux**. C'est
  exactement l'écart que BENCHMARKS.md attribue à `svd` et `lstsq`.
- [Van Zee, van de Geijn, Quintana-Ortí et Elizondo, ACM TOMS 39(1), 2012](https://www.cs.utexas.edu/~flame/pubs/CondensedFormTOMS.pdf)
  généralisent la fusion à la réduction tridiagonale (`eigh`) et de Hessenberg (utile pour le futur `eig` non
  symétrique).

### 5.3 L'itération QR restructurée

- [Van Zee, van de Geijn et Quintana-Ortí, ACM TOMS 40(3), 2014](https://repositori.uji.es/xmlui/handle/10234/123806)
  appliquent les rotations de Givens de plusieurs balayages par vagues, fusionnées et vectorisées. L'accumulation des
  vecteurs approche alors la vitesse d'un GEMM. La méthode devient **compétitive avec D&C et MRRR**, avec un espace
  de travail en O(n), et c'est selon ses auteurs la SVD bidiagonale la plus rapide connue. Elle s'applique
  directement à `eigen_method::tridiagonal_ql` et `svd_method::bidiagonal_qr`.

### 5.4 Divide-and-conquer accéléré par matrices HSS (atypique)

- Dans une fusion de D&C, la matrice de vecteurs propres est de **type Cauchy**, donc de rang faible hors de la
  diagonale. [Li, Liao et al., NLAA 2016](https://arxiv.org/abs/1510.04591) l'approchent par une matrice HSS, pour un
  coût en O(n² r) au lieu de O(n³). Ils rapportent jusqu'à **plus de 6× MKL** sur de grandes matrices qui déflatent
  peu ; [PSDC (Liao et al., 2020)](https://arxiv.org/abs/2008.01990) obtient 1,4 à 1,6× ScaLAPACK. L'approximation se
  règle à ε près, mais la stabilité inverse est à démontrer pour garder le contrat D.

### 5.5 Divide-and-conquer spectral par décomposition polaire

- QDWH ([Nakatsukasa, Bai et Gygi, SIMAX 31(5), 2010](https://doi.org/10.1137/090774999) ;
  [Nakatsukasa et Higham, SISC 35(3), 2013](https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/pub13.pdf)) et
  Zolo-PD (Nakatsukasa et Freund, SIAM Review 58(3), 2016) n'utilisent que QR, Cholesky et GEMM, avec une stabilité
  inverse démontrée. Ils font plus de flops, mais tout est en BLAS3 et massivement parallèle.
- QDWH-SVD atteint jusqu'à 3× ScaLAPACK sur Haswell et 8× sur KNL
  ([Sukkari, Ltaief, Esposito et Keyes, ACM TOMS, 2019](https://unpaywall.org/10.1145%2F3309548)). Sur 8 à 10 cœurs,
  c'est spéculatif ; la méthode prend son sens avec beaucoup de cœurs ou un débit FP32 ou bf16 élevé, à combiner
  avec §4.

### 5.6 Algorithmes récursifs

- [ReLAPACK (Peise et Bientinesi, ACM TOMS 44(2), 2017)](https://arxiv.org/abs/1602.06763) est un remplaçant
  récursif de la plupart des algorithmes bloqués de LAPACK. Il demande **peu de réglage** et dépasse souvent OpenBLAS
  et MKL. Les fondements remontent à Toledo (SIMAX 18(4), 1997) pour la LU récursive et à Gustavson (IBM J. R&D
  41(6), 1997), puis Elmroth, Gustavson, Jonsson et Kågström (SIAM Review 46(1), 2004).
- La récursion règle la taille de bloc d'elle-même : aux tailles moyennes (256 à 512), où les panneaux fixes de 32
  ou 64 colonnes sous-performent, c'est un levier simple et élégant.

### 5.7 Inverse et cas symétrique défini positif

- La **Gauss-Jordan par blocs** (Quintana-Ortí, Quintana-Ortí, Sun et van de Geijn, SISC 22(5), 2001 ;
  [Benner et al., 2013](https://csc.mpi-magdeburg.mpg.de/preprints/2012/02/)) inverse en un seul balayage, avec le
  même nombre de flops que LU suivi de l'inversion triangulaire, mais en BLAS3 mieux équilibré. Elle s'est montrée la
  plus performante sur les plateformes hybrides.
- Pour une matrice symétrique définie positive, Bientinesi, Gunter et van de Geijn (ACM TOMS 35(1), 2008) dérivent
  des familles d'algorithmes d'inversion en un balayage. La méthode FLAME produit toutes les variantes ; on choisit
  la meilleure par architecture.

### 5.8 LU : éviter le panneau séquentiel

- **RBT** : une transformation papillon aléatoire rend le pivotage inutile, et la LU devient entièrement BLAS3
  (Parker, 1995 ; [Baboulin, Dongarra, Herrmann et Tomov, ACM TOMS 39(2), 2013](https://impact.ornl.gov/en/publications/accelerating-linear-system-solutions-using-randomization-techniqu/)).
  La généralisation aux tailles quelconques est dans SLATE ([Lindquist, Luszczek et Dongarra, ACM TOMS 2024](https://netlib.org/utk/people/JackDongarra/PAPERS/butterfly-toms-2024.pdf)).
  À combiner avec le raffinement (§4.1) : contrat **C**.
- **CALU** : le pivotage par tournoi ([Grigori, Demmel et Xiang, SIMAX 32(4), 2011](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-29.html))
  réduit les communications du panneau ; une variante plus robuste existe (CALU_PRRP).
- **Look-ahead et graphe de tâches** : PLASMA (Buttari, Langou, Kurzak et Dongarra, Parallel Computing 35(1), 2009)
  et la LU malléable (Catalán et al., IEEE Access, 2019) recouvrent le panneau par la mise à jour de traîne. Pour
  NumericAL, il faudrait dépasser le modèle fork-join de `parallel_for` par un petit ordonnanceur de tâches à
  dépendances, ce qui reste faisable avec `std::thread`.

---

## 6. Randomisation (RandNLA)

Synthèse orientée logiciel : [Murray et al., *Randomized numerical linear algebra: a perspective on the field with an
eye to software* (2023)](https://arxiv.org/abs/2302.11474). RandBLAS en est une référence de conception, en C++
header-only, mais elle dépend de BLAS++ et Random123.

### 6.1 `lstsq` très rectangulaire (m ≫ n)

- [Blendenpik (Avron, Maymounkov et Toledo, SISC 32(3), 2010)](https://dspace.mit.edu/bitstream/1721.1/60954/2/Avron-2010-BLENDENPIK_%20SUPERCHA.pdf)
  est environ **4× plus rapide que `dgels` de LAPACK** sur les grandes matrices hautes, mais plus lent sur les petites
  ou presque carrées. [LSRN (Meng, Saunders et Mahoney, SISC 2014)](https://arxiv.org/abs/1109.5981) suit la même
  approche.
- **Mise en garde** : sous sa forme courante, le *sketch-and-precondition* est numériquement instable
  ([Meier, Nakatsukasa, Townsend et Webb, 2023](https://ar5iv.labs.arxiv.org/html/2302.07202)).
- **Solution** : SPIR et FOSSILS sont **stables au sens inverse**, aussi précis qu'une QR et aussi rapides que les
  solveurs randomisés ([Epperly, Meier et Nakatsukasa, CPAM 2026](https://arxiv.org/abs/2406.03468)). Le *iterative
  sketching* est stable au sens direct (Epperly, SIMAX 2024).

### 6.2 QR haute et étroite

- CholeskyQR2 (Fukaya et al., 2014) et **shifted CholeskyQR3**
  ([Fukaya, Kannan, Nakatsukasa, Yamamoto et Yanagisawa, SISC 42(1), 2020](https://arxiv.org/abs/1809.11085))
  n'utilisent que SYRK, Cholesky et TRSM : du BLAS3 pur, stable jusqu'à κ ≈ ε⁻¹ avec le décalage.
- Variantes randomisées : [Higgins, Szyld, Boman et Yamazaki (2023)](https://arxiv.org/abs/2309.05868) ;
  [Householder randomisée (Grigori et Timsit, 2024)](https://arxiv.org/abs/2405.10923), qui hérite de la stabilité de
  Householder pour la moitié du coût ; TSQR (Demmel, Grigori, Hoemmen et Langou, SISC 34(1), 2012).

### 6.3 Révéler le rang sans SVD

- La QR à pivotage randomisé **HQRRP** (Martinsson, Quintana-Ortí, Heavner et van de Geijn, SISC 39(2), 2017) choisit
  les pivots sur un sketch : elle est BLAS3, contrairement à `dgeqp3`, qui reste à moitié BLAS2. Elle sert à
  `matrix_rank` et au repli de §3.4.

### 6.4 Valeurs propres partielles

- Le GMRES et le Rayleigh-Ritz « sketchés » de [Nakatsukasa et Tropp, SIMAX 2024](https://arxiv.org/abs/2111.00113)
  annoncent 10 à 100× sur MATLAB pour les problèmes partiels. Hors du périmètre actuel (spectre complet), ils seraient
  utiles pour une future API « k valeurs propres ».

---

## 7. Mémoire et données : einsum, transposition, lots

- **Transposition** (300 fois plus lente que NumPy dans l'historique) : [HPTT (Springer, Su et Bientinesi, ARRAY
  2017)](https://arxiv.org/abs/1704.04374) découpe toute permutation de tenseur en boucles autour d'un micro-noyau, à
  la manière de BLIS. Il atteint une bande passante comparable à SAXPY et se règle de lui-même à l'exécution.
- **Contraction sans transposition** : [TBLIS (Matthews, SISC 40(1), 2018)](https://arxiv.org/abs/1607.00291) et
  GETT (Springer et Bientinesi, ACM TOMS 44(3), 2018) intègrent la permutation au packing. `einsum` atteindrait alors
  la vitesse du GEMM pour toute contraction, pas seulement `"ij,jk->ik"`.
- **Chemin de contraction** pour plus de deux opérandes : opt_einsum (Smith et Gray, JOSS 3(26), 2018).
- **Agencements récursifs** : les agencements de Morton et les agencements récursifs par tuiles
  ([Chatterjee, Lebeck, Patnala et Thottethodi, IEEE TPDS 13(11), 2002](https://researcher.ibm.com/publications/recursive-array-layouts-and-fast-matrix-multiplication))
  accélèrent le produit standard de 1,2 à 2,5×, à condition que la récursion s'arrête à des tuiles en ordre canonique.
- **Lots de petites matrices** (pour une future API par lots) : la disposition **compacte** entrelace les matrices
  selon la largeur SIMD ([Kim et al., SC17](https://www.sandia.gov/research/publications/details/designing-vector-friendly-compact-blas-and-lapack-kernels-2017-11-12/)).
  Elle atteint jusqu'à 14×, 45× et 27× face à des boucles OpenMP autour de DGEMM, DTRSM et DGETRF ; oneMKL l'a
  adoptée.

---

## 8. Horizon, et pistes écartées

**Horizon** (recherche fondamentale) :

- *Fast linear algebra is stable* (Demmel, Dumitriu et Holtz, Numer. Math. 108, 2007) : toute l'algèbre linéaire
  dense en O(n^ω), de façon stable.
- *Pseudospectral shattering* (Banks, Garza-Vargas, Kulkarni et Srivastava, 2020) : diagonaliser une matrice non
  symétrique en temps quasi matriciel. Théorique, mais pertinent pour le futur `eig` non symétrique.

**Écartées**, avec leur raison :

| Piste | Raison |
| --- | --- |
| Inverse FP32 raffinée par Newton–Schulz | après une itération, `‖I − AX‖ ≈ (κ·u₃₂)²`, qui n'atteint jamais `κ·u₆₄` (il faudrait κ < 0,03) ; deux itérations coûtent 8 n³ flops contre 2 n³ pour l'inversion directe |
| Produits approchés (MADDNESS, Blalock et Guttag 2021) | résultats inexacts, hors contrat |
| Produit scalaire de Winograd (1968) | réduit les multiplications au prix d'additions ; aucun gain avec la FMA |
| 3M par défaut pour les complexes | borne d'erreur affaiblie sur la partie imaginaire ; opt-in seulement (§2.5) |
| Morton jusqu'à l'élément | contre-productif (Chatterjee et al.) ; s'arrêter à la tuile |
| Ozaki sur Apple SME | INT8 et BF16 relativement lents sur M4 |

---

## 9. Choix de standards proposés

### 9.1 Le contrat numérique des variantes rapides

| Option | Atouts | Détriments |
| --- | --- | --- |
| (a) Une énumération par opération (`eigen_method`, `svd_method`, demain `gemm_algorithm`, `solve_precision`) | explicite, déjà en place, testable | l'utilisateur doit connaître les algorithmes |
| (b) Une politique globale (`numerical::numeric_policy` : `classic_bits`, `deterministic`, `fast`) bornant ce que le dispatch automatique peut choisir, réglable par variable d'environnement comme `NUMERICAL_GEMM` | un seul bouton ; reflète le contrat du README | état global ; à documenter par opération |
| (c) Le choix automatique à la manière de LAPACK (`dsgesv` mixte avec repli) | gains sans rien changer côté appelant | surprend qui compare les bits avec NumPy |

**Recommandation : (a) + (b).** Une énumération par appel, et une politique globale par défaut à `deterministic`,
qui garantit les mêmes bits quel que soit le nombre de fils, comme aujourd'hui. Les pistes **B** et **D** deviennent
admissibles d'office ; **C** l'est seulement avec un repli prouvé ; **O** jamais sans demande explicite.

### 9.2 Comment produire les noyaux

| Option | Atouts | Détriments |
| --- | --- | --- |
| Intrinsèques et `__attribute__((target))` avec `cpuid` (actuel) | contrôle total, zéro outil | une tuile écrite à la main par jeu d'instructions |
| `target_clones` (GCC et Clang) | dispatch automatique par ifunc | ELF seulement ; pas de MSVC ; pas de contrôle des tuiles |
| Génération hors ligne (Exo) dont la sortie est versionnée | tuiles et bords générés, vérifiables | un outil de développement à installer (pas à l'usage) |
| JIT header-only | spécialisation aux tailles réelles | W^X, environnements durcis ; surface de risque |
| `std::simd` (C++26) | standard, portable, couvrira MSVC | support jeune ; performances inégales |

**Recommandation** : garder l'actuel pour les tuiles chaudes, **générer hors ligne** les nouvelles cibles (SME, bords,
petites tailles fixes), adopter `std::simd` pour le chemin portable quand les trois compilateurs l'auront, et réserver
le JIT à une expérience derrière un drapeau de compilation.

### 9.3 Le critère de repli en précision mixte

| Option | Atouts | Détriments |
| --- | --- | --- |
| Nombre maximal d'itérations (`dsgesv` : 30) | simple | ne mesure pas la qualité |
| Estimation du conditionnement avant de commencer | évite un travail inutile | coûte O(n²) par estimation ; estimateur approximatif |
| **Erreur inverse componentwise (Oettli–Prager) à chaque itération** | garantit le contrat : la même erreur inverse que FP64 | un produit matrice-vecteur FP64 par itération |

**Recommandation** : l'erreur inverse comme critère d'acceptation, plus un plafond d'itérations, comme `dsgesv`.

---

## 10. Feuille de route proposée

| Phase | Contenu | Pourquoi d'abord |
| --- | --- | --- |
| **1. Gains sûrs** | `lstsq` par polyalgorithme (§3.4) ; complexes en 1M (§2.5) ; `kc`/`mc` par jeu d'instructions, préchargement, grandes pages (§2.1) ; `symv` vectorisé (§5.1) | effort faible ; contrat **B** ou **D** ; vise les écarts mesurés |
| **2. Restructuration** | fusion BLAS 2.5 (§5.2) ; ordonnanceur par forme et région persistante (§2.2) ; LU, Cholesky et `trtri` récursifs (§5.6) ; Gauss-Jordan pour `inverse` (§5.7) ; raffinement mixte pour `solve` (§4.1) | effort moyen ; ferme la plupart des écarts de 1,2 à 2,9× |
| **3. Nouveaux matériels** | SME par ACLE (§2.4) ; Strassen opt-in (§3.1) ; réduction en deux étapes pour les valeurs seules (§5.1) | pérennité et grandes tailles |
| **4. Recherche** | Ozaki-II INT8 sur AMX et mode reproductible inter-noyaux (§4.4) ; Ogita–Aishima sur M4 et suivants (§4.2) ; MRRR mixte (§4.3) ; D&C par HSS (§5.4) ; QDWH (§5.5) ; FOSSILS (§6.1) ; RBT (§5.8) | potentiellement de rupture, mais à valider |

Chaque étape passe par les portes existantes : `make fuzz-linalg` sur chaque chemin, `make bench-linalg` sur les trois
machines de référence, et une couverture de 100 %.
