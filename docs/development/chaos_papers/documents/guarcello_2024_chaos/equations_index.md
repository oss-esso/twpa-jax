# Numbered equation index — Driving a Josephson Traveling Wave Parametric Amplifier into chaos: effects of a non-sinusoidal current-phase relation

This index follows the equation numbering of the supplied source and contains one entry per expected numbered equation. It is a retrieval aid, not a re-typeset edition. The complete source text is in `source_fulltext_layout.txt` and `source_pages.jsonl`. Native PDF extraction can garble fractions, radicals, superscripts or Greek glyphs, especially in the 1980s papers; use the cited PDF page and printed equation label as authoritative for exact typography. Unnumbered mathematics is preserved in the full-text files.

Numbered equations indexed: **68**.

## Eq. (1) — PDF page 2

```text
is very effective in capturing dynamic spectral proper-          written as
ties and the geometrical structure of phase space, respec-
tively. While alternative methods, like the study of Lya-                         ℏ d2 φn    1 ℏ dφn
                                                                      IJ,n = CJ         2
                                                                                          +          + Ic sin(φn ).        (1)
punov exponents, can offer insights into JJ stability and                         2e dt     RJ 2e dt
chaotic behavior [24–27], FT and PSs provide a quite             Here, we assume that each JJ has a critical current
intuitive understanding of the complex interactions and          Ic = 2 µA, a quasiparticle resistance RJ = 20 kΩ, and
```

## Eq. (2) — PDF page 2

```text
junction, i.e., the skewness of the current-phase relation                 dt , one obtains a current-voltage profile charac-
                                                                 teristic for a nonlinear inductor, V = LJ0 (φ)dI/dt, where
(CPR), impacts on the device performances. In Sec. VI,
conclusions are drawn.                                                                          ℏ      1
                                                                                    LJ0 (φ) =                              (2)
                                                                                                2e Ic cos(φ)
                      II.   MODEL                                denotes the intrinsic Josephson inductance. Considering
                                                                 a non-zero DC bias current, Ibias , the effective Josephson
```

## Eq. (3) — PDF page 5

```text
Ppump = −54.5 dBm. We achieve Gain ∼ 8 dB in the              the phase matching across the rf–SQUID arms leads to
νsign ∈ [6−8] GHz frequency bandwidth. Along this gain        the condition
profile, periodic variations, usually referred to as “rip-
ples”, are prominently observed (refer to the detailed view                                Lg Ig
                                                                                      2π         = φ.                    (3)
in the inset); each ripple is associated in this case with                                  Φ0
small gain variations of ∼ 0.2 dB. Common to all sys-
                                                              When φ = 2πn, with n = 0, ±1, ±2 . . . , the current
```

## Eq. (4) — PDF page 6

```text
and 4WM nonlinear coefficients [14], respectively given              the strongly oscillating behaviour of these quantities, it
by                                                                   is more convenient to extract only the mean value and
                                                                     the standard deviation of β(t) and γ(t) from each time
           βL                             βL
    β=        sin(φdc )    and       γ=      cos(φdc ),     (4)      evolution. In this way, it is possible to plot both β̄, σβ ,
            2                              6                         γ̄, and σγ as Ppump and Ibias vary, see Fig. 4. As Ppump
where βL = 2πLIc /Φ0 is the screening parameter of the               increases, β̄ (red curve) remains close to zero, but σβ
rf-SQUID (in our case, βL ≃ 0.74) and φdc = 2πΦdc /Φ0 ,              (yellow curve) tends to increase, due to the increasing
```

## Eq. (5) — PDF page 7

```text
contributions as the bias current is varied. Specifically,           tor–superconductor (SNS) junction, showcasing an in-
within the orange–shaded area, β̄ increases significantly,           trinsically non-sinusoidal CPR, like [55–57]
while its standard deviation reaches a sort of plateau; in-
sted, γ̄ decreases to almost zero, while its standard de-                                          τ sin φ
                                                                                        Iτ (φ) = p             ,                (5)
viation increases. The average values of β and γ seem                                           2 1 − τ sin2 φ
to follow similar patterns in the orange and blue regions,
but with opposite trends. This is due to the 2π–phase                where τ ∈ [0, 1] is the transparency. Its limit values lead
```

## Eq. (6) — PDF page 8

```text
imum value, that is                                                                                                 Ibias=17.1µ A
                                                                                    0
                                      √                                            -58          -56         -54   -52     -50
                            sin φ 1 − 1 − τ
                i (φ, τ ) =    p               .           (6)                                          Ppump [dBm]
                              2 1 − τ sin2 φ
   In the following, we assume again a critical current            FIG. 6. (a) and (b), Gain versus pump power level, Ppump ,
equal to Ic = 2 µA, just like in the previous section, and         choosing different junction transparency, at Ibias = 0 and
```

## Eq. (A1) — PDF page 10

```text
limit, where the current-phase relation tends to become           to get the solution of the setup in Fig. 1.
increasingly skewed. Furthermore, at these low temper-              We start considering a generic mesh labeled with “n”.
atures, the reduction in thermal noise, combined with             The current balance at node n is
the presence of broadband noise, can enhance chaotic
dynamics and increase stochasticity, e.g., see Ref. [42].                               In = q̇n + In+1                   (A1)
These factors underscore the need for a thorough inves-
tigation of chaotic effects in JTWPAs to optimize their           while the current balance at node nJL is
low–temperature performance.
```

## Eq. (A2) — PDF page 10

```text
dynamics and increase stochasticity, e.g., see Ref. [42].                               In = q̇n + In+1                   (A1)
These factors underscore the need for a thorough inves-
tigation of chaotic effects in JTWPAs to optimize their           while the current balance at node nJL is
low–temperature performance.
  In conclusion, our results are particularly relevant                                  In = IJ,n + IL,n                  (A2)
```

## Eq. (A3) — PDF page 11

```text
                                                                                In−1 − In     ℏ d2 φn   In − In+1
                                                                                  −        +          +           = 0
                ℏ d2 φn    1 ℏ dφn                                                 Cn−1       2e dt 2      Cn
  IJ,n = CJ,n         2
                        +            + Ic,n sin φn        (A3)
                2e dt     RJ,n 2e dt                                                                      ℏ d2 φn
                                                                                                     
                                                                       In−1   In+1          1       1
```

## Eq. (A4) — PDF page 11

```text
                                                                    normalization to 2e
                                                                                     ℏ )

                                 1 ℏ
                       IL,n =           φn .              (A4)              eJ,n = CJ,n ℏ
                                                                            C                               e−1 = R−1 ℏ
                                                                                                            R                 (A7)
                                Lg,n 2e                                                                       J,n   J,n
```

## Eq. (A5) — PDF page 11

```text

                                                                                 d2 φn    1 dφn                    1
                  qn−1   ℏ dφn   qn                                  In = C
                                                                          eJ,n         +         + Ic,n sin φn +      φn (A9)
                −      +       +    =0                    (A5)                    dt2    RJ,n dt                 Lg,n
                  Cn−1   2e dt   Cn
                                                                                         e                       e
```

## Eq. (A6) — PDF page 11

```text
                                                                     −Cn− In−1 + 1 + Cn− In + C
                                                                                        
                                                                                                  dt2
                    q̇n−1   ℏ d2 φn   q̇n
                −         +         +     = 0.            (A6)      Inserting Eq. (A9) in Eqs. (A10)
                    Cn−1    2e dt2    Cn

                                    "                                                                      #
```

## Eq. (A7) — PDF page 11

```text

                                 1 ℏ
                       IL,n =           φn .              (A4)              eJ,n = CJ,n ℏ
                                                                            C                               e−1 = R−1 ℏ
                                                                                                            R                 (A7)
                                Lg,n 2e                                                                       J,n   J,n
                                                                                        2e                              2e
                                                                              −1    −1 ℏ                            ℏ
```

## Eq. (A8) — PDF page 11

```text
                                                                                        2e                              2e
                                                                              −1    −1 ℏ                            ℏ
                                                                            Lg,n = Lg,n
                                                                            e                               Cn = Cn
                                                                                                            e                 (A8)
                                                                                        2e                          2e
The voltage drop in the mesh “n”, with n ∈ [1, ..., N ], is
                                                                    and inserting Eqs. (A3)-(A4) in Eqs. (A2), one obtains
```

## Eq. (A9) — PDF page 11

```text
                                                                    and inserting Eqs. (A3)-(A4) in Eqs. (A2), one obtains

                                                                                 d2 φn    1 dφn                    1
                  qn−1   ℏ dφn   qn                                  In = C
                                                                          eJ,n         +         + Ic,n sin φn +      φn (A9)
                −      +       +    =0                    (A5)                    dt2    RJ,n dt                 Lg,n
                  Cn−1   2e dt   Cn
                                                                                         e                       e
```

## Eq. (A10) — PDF page 11

```text
                                                                     −Cn− In−1 + 1 + Cn− In + C
                                                                                        
                                                                                                  dt2
                    q̇n−1   ℏ d2 φn   q̇n
                −         +         +     = 0.            (A6)      Inserting Eq. (A9) in Eqs. (A10)
                    Cn−1    2e dt2    Cn

                                    "                                                                      #
```

## Eq. (A11) — PDF page 11

```text
                                    "                                                                      #
                                               2
                                             d   φ n−1       1    dφ n−1                           1
                          −Cn−        C
                                      eJ,n−1           +                 + Ic,n−1 sin φn−1 +           φ                     (A11)
                                                 dt2     ReJ,n−1 dt                            e g,n−1 n−1
                                                                                               L
                                    "                    !                                              #
```

## Eq. (A12) — PDF page 11

```text
                                                    en     d   φn     1  dφ n                     1
                    + 1 + Cn−
                                
                                        C
                                        eJ,n +                    +            + Ic,n sin φn +       φ                       (A12)
                                                 1 + Cn−    dt2     ReJ,n dt                   Le g,n n
                                    "                                                                      #
                                              2
```

## Eq. (A13) — PDF page 11

```text
                                    "                                                                      #
                                              2
                                             d   φ n+1      1     dφn+1                           1
                                −     C
                                      eJ,n+1           +                 + Ic,n+1 sin φn+1 +           φ     =0              (A13)
                                                dt2      ReJ,n+1 dt                            e g,n+1 n+1
                                                                                               L
```

## Eq. (A14) — PDF page 11

```text
   Discretization − For the numerical integration of pre-           partial derivatives can be expressed as
vious equations, the time was divided into many short
time intervals k = ∆t = tmax /M , where tmax and M                                  ∂φn     φm+1 − φm−1
                                                                                          ≃ n       n
                                                                                                                             (A14)
are the observation time and the number of intervals,                                ∂t          2k
respectively. The partial derivatives are approximated                              ∂ 2 φn   φm+1
                                                                                              n   − 2φm    m−1
```

## Eq. (A15) — PDF page 11

```text
are the observation time and the number of intervals,                                ∂t          2k
respectively. The partial derivatives are approximated                              ∂ 2 φn   φm+1
                                                                                              n   − 2φm    m−1
                                                                                                      n + φn
using the Euler formalism. The phase φn (t) is labeled by                                  ≃                   .             (A15)
                                                                                     ∂t2             k2
φmn = φn (mk), where n and m are the discrete mesh and
                                       → time index
```

## Eq. (A16) — PDF page 11

```text
             eJ,n                              1                                    φmn
                  φm+1   − 2φm      m−1
                                                     φm+1 − φm−1   + Ic,n sin φm
                                                                
        In ≃    2  n         n + φn       +           n      n                 n +                                           (A16)
              k                             2R
                                             eJ,n k                                 L
                                                                                    e g,n
```

## Eq. (A17) — PDF page 11

```text
                                    !                                     !                            !
              m+1  CeJ,n      1             2C
                                             eJ,n
                                                    m          m    φmn         m−1    CeJ,n     1
        In ≃ φn          +            + − 2 φn + Ic,n sin φn +              + φn             −           ,                   (A17)
                     k2     2R
                             eJ,n k          k                      L
                                                                    e g,n                k2    2R
```

## Eq. (A18) — PDF page 11

```text

which becomes                                                       after defining the quantities
               In ≃ αn+ φm+1
                         n   + fnm + αn− φm−1
                                          n              (A18)                        C
                                                                                      eJ,n     1
                                                                             αn± =       2
                                                                                           ±                                 (A19)
```

## Eq. (A19) — PDF page 11

```text
                         n   + fnm + αn− φm−1
                                          n              (A18)                        C
                                                                                      eJ,n     1
                                                                             αn± =       2
                                                                                           ±                                 (A19)
                                                                                       k     2R
                                                                                              eJ,n k
                                                                                                       !
```

## Eq. (A20) — PDF page 11

```text
                                                                                                       !
                                                                                         1     2C
                                                                                                eJ,n
                                                                             fnm =           −             φm             m
                                                                                                            n + Ic,n sin φn . (A20)
                                                                                       L
                                                                                       e g,n    k2
```

## Eq. (A21) — PDF page 12

```text
                                                                                                                            (A23)
                           C
                           e“n”      1
                   en± =
                   α            ±                             (A21)        Similarly, after discretization of the terms in square
                            k2    2RJ,n k
                                   e                                       brackets in Eqs. (A11)-(A13), one obtains:
```

## Eq. (A22) — PDF page 12

```text
                              !
               1      2C
                       e“n”
     fenm =         −             φm             m
                                   n + Ic,n sin φn ,          (A22)
              L
              e g,n    k2
```

## Eq. (A23) — PDF page 12

```text
                                                                            e“n” =       C
                                                                                         eJ,n +                 =
                                                                                                              CJ,n +             .
                                                                                                  1 + Cn−         2e Cn−1 + Cn
                                                                                                                            (A23)
                           C
                           e“n”      1
                   en± =
```

## Eq. (A24) — PDF page 12

```text
                                                m
                                                   + Cn− αn−1 φm−1        −
                                                                                           φm−1
                                                                            + m−1
                                                               n−1 − 1 + Cn αen φn  + αn+1  n+1                                            (A24)

that becomes
```

## Eq. (A25) — PDF page 12

```text

    an,1 φm+1         m+1
          n−1 + an,2 φn   + an,3 φm+1         m          em         m           m−1         m−1
                                  n+1 = bn,1 fn−1 + bn,2 fn + bn,3 fn+1 + cn,1 φn−1 + cn,2 φn   + cn,3 φm−1
                                                                                                        n+1                                (A25)

defining the coefficients:
                                             +                                                           +
```

## Eq. (A26) — PDF page 12

```text
                                             +                                                           +
                               an,1 = −Cn− αn−1              an,2 = (1 + Cn− ) α
                                                                               en+            an,3 = −αn+1
                                        −                                    −
                               bn,1 = Cn                     bn,2 = − (1 + Cn )               bn,3 = 1                                     (A26)
                                           −                                                           −
                               cn,1 = Cn− αn−1                                   en−
                                                             cn,2 = − (1 + Cn− ) α            cn,3 = αn+1
```

## Eq. (A27) — PDF page 12

```text
   The current balances and voltage drops at the leftmost
mesh in left panel of Fig. 1 gives
                                                                                      V̇i    q̇0    V̇i   Ii + Ibias − I1
               q0                                                               I˙i =     −       =     −                                  (A28)
  Vi = Ii Ri +             Ii + Ibias = q̇0 + I1     (A27)                            Ri    Ri Ci   Ri         Ri Ci
               Ci
             ℏ d2 φ1    1 ℏ dφ1                      1
  I1 = CJ,1          +              + Ic,1 sin φ1 +      φ1
```

## Eq. (A28) — PDF page 12

```text
side of the circuit including the voltage generator Vi (t).
   The current balances and voltage drops at the leftmost
mesh in left panel of Fig. 1 gives
                                                                                      V̇i    q̇0    V̇i   Ii + Ibias − I1
               q0                                                               I˙i =     −       =     −                                  (A28)
  Vi = Ii Ri +             Ii + Ibias = q̇0 + I1     (A27)                            Ri    Ri Ci   Ri         Ri Ci
               Ci
             ℏ d2 φ1    1 ℏ dφ1                      1
```

## Eq. (A29) — PDF page 12

```text

                                                       !                                                                   !
                                                                        2
                                      V̇i                         eJ,1 d φ1 + 1 dφ1 + Ic,1 sin φ1 + 1 φ1
                   I˙i = −ωi Ii +         − ωi Ibias       + ωi   C                                                            .           (A29)
                                      Ri                                dt2  ReJ,1 dt              L
                                                                                                   e g,1
```

## Eq. (A30) — PDF page 12

```text
                                            2C
                                             eJ,1
                                                  m           m   φm1         m−1     C
                                                                                      eJ,1     1
                                         + − 2 φ1 + Ic,1 sin φ1 +          + φ1            −                                               (A30)
                                             k                    L
                                                                  e g,1                k2    2R
                                                                                              eJ,1 k
```

## Eq. (A31) — PDF page 12

```text
                                                                                         e g,1     k2
                                                                                              !
                                                        I m−1                  C
                                                                               eJ,1     1
                                                       + i    + φ1m−1               −           .                                          (A31)
                                                        2kωi                    k2    2R
                                                                                       eJ,1 k
```

## Eq. (A32) — PDF page 13

```text
and
                           Iim+1                          I m−1                            
                                 − φm+1
                                    1   α1+ = −Iim + f1m + i    + φm−1
                                                                   1   α1− + Ci V̇im − Ibias ,                             (A32)
                           2kωi                            2kωi
and finally
                                      h                                                 i
```

## Eq. (A33) — PDF page 13

```text
                                                      + f m
                                                          1 + φ 1  α1 +   C V̇
                                                                           i i
                                                                               m
                                                                                 − Ibias    (2kωi ) .                      (A33)
```

## Eq. (A34) — PDF page 13

```text

 The current balances at nodes and voltage drops at the                In other words, we are replacing In−1 with (Ii + Ibias ) in
mesh labeled with “1” in the left panel of Fig. 1 gives                Eq. (A7), so that

                    Ii + Ibias − q̇0 − I1 = 0               (A34)
                        I1 − I2 − q˙0 = 0                   (A35)
                                                                                                               2
                               2
```

## Eq. (A35) — PDF page 13

```text
 The current balances at nodes and voltage drops at the                In other words, we are replacing In−1 with (Ii + Ibias ) in
mesh labeled with “1” in the left panel of Fig. 1 gives                Eq. (A7), so that

                    Ii + Ibias − q̇0 − I1 = 0               (A34)
                        I1 − I2 − q˙0 = 0                   (A35)
                                                                                                               2
                               2
                                                                                                           e1 d φ1 −I2 = 0, (A38)
```

## Eq. (A36) — PDF page 13

```text
                                                                           −C1− (Ii +Ibias )+ 1 + C1− I1 + C
                                                                                                     
                   − Cq̇00 + 2e
                             ℏ d φ1   q̇1
                                dt2 + C1 = 0                (A36)
                                                                                                               dt2
from which
          (Ii + Ibias ) − I1   ℏ d2 φ1   q̇1
```

## Eq. (A37) — PDF page 13

```text
                                dt2 + C1 = 0                (A36)
                                                                                                               dt2
from which
          (Ii + Ibias ) − I1   ℏ d2 φ1   q̇1
      −                      +         +     = 0.           (A37)      which, making all terms explicit, becomes
                 C0            2e dt2    C1

                                                    "                      !                                           #
```

## Eq. (A38) — PDF page 13

```text
                    Ii + Ibias − q̇0 − I1 = 0               (A34)
                        I1 − I2 − q˙0 = 0                   (A35)
                                                                                                               2
                               2
                                                                                                           e1 d φ1 −I2 = 0, (A38)
                                                                           −C1− (Ii +Ibias )+ 1 + C1− I1 + C
                                                                                                     
                   − Cq̇00 + 2e
```

## Eq. (A39) — PDF page 13

```text
                                                    "                                                #
                                                             2
                                                           d   φ2    1  dφ 2                  1
                                              −       C
                                                      eJ,2        +          + Ic,2 sin φ2 +      φ = 0.                   (A39)
                                                            dt2     eJ,2 dt
                                                                    R                        e g,2 2
                                                                                             L
```

## Eq. (A40) — PDF page 13

```text
              −C1− [Iim + Ibias ] + 1 + C1−                      e1− φm−1
                                                        + fe1m + α         − α2+ φm+1 + f2m + α2− φ2m−1 = 0.
                                                                                                       
                                                 α
                                                 e1 φ1                1           2                                        (A40)

By slightly manipulating this equation, one obtains
```

## Eq. (A41) — PDF page 13

```text
      1 + C1− α                    = C1− Iim − 1 + C1− fe1m + f2m − 1 + C1− α      + α2− φm−1 + C1− Ibias
              + m+1
                       − α2+ φm+1
                                                                           + m−1
              e1 φ1            2                                            e 1 φ1        2                                (A41)

that can be recast in a compact form as
```

## Eq. (A42) — PDF page 13

```text
                    a1,2 φm+1
                          1   + a1,3 φm+1
                                      2   = b1,1 Iim + b1,2 fe1m + b1,3 f2m + c1,2 φm−1
                                                                                    1   + c1,3 φm−1
                                                                                                2   + C1− Ibias            (A42)

by defining the coefficients
```

## Eq. (A43) — PDF page 13

```text

                                                        a1,2 = 1 + C1− α
                                                                       +
                                      a1,1 = 0                          e1           a1,3 = −α2+
                                      b1,1 = C1−        b1,2 = − 1 + C1−             b1,3 = 1                             (A43)
                                      c1,1 = 0          c1,2 = − 1 + C1− α e1−        c1,3 = α2− .
```

## Eq. (A44) — PDF page 13

```text

Comparing with the coefficients in Eq. (A26), it means                 most mesh of Fig. 1 gives
to impose α0± = 0.
                                                                                                    qN     qℓ
                                                                       IN = Ibias + q̇N + q̇ℓ        ,  =     + q̇ℓ Rℓ (A44)
                                                                                                    CN     Cℓ
                                                                                       2
                                                                       IN        eJ,N d φN + 1 dφN + Ic,N sin φN + 1 φN .
```

## Eq. (A45) — PDF page 14

```text
                                                                                                                       !
                                                                    2
                                
                    ˙         CN                              eJ,N d φN + 1 dφN + Ic,N sin φN + 1 φN
             CN Rℓ Iℓ = − 1 +      Iℓ − Ibias +               C                                                            ,   (A45)
                              Cℓ                                    dt2  eJ,N dt
                                                                         R                     L
                                                                                               e g,n
```

## Eq. (A46) — PDF page 14

```text
                                            + − 2 φm               m
                                                     N + Ic,N sin φN +
                                                                         N
                                                                               + φm−1
                                                                                  N           −           ,                    (A46)
                                                k                      L
                                                                       e g,n              k2    2R
                                                                                                 eJ,N k
```

## Eq. (A47) — PDF page 14

```text
                                                                                                !
                                                                             C
                                                                             eJ,N      1                CN Rℓ m−1
                                                                  +   φm−1
                                                                       N          −                 +        I    − Ibias .    (A47)
                                                                              k 2
                                                                                    2R
                                                                                     eJ,N k              2k ℓ
```

## Eq. (A48) — PDF page 14

```text


  The output voltage is finally equal to                                from which
                        m+1
                      Vout  = Iℓm+1 Rℓ .                  (A48)

   The balances of the currents at nodes and voltage                            IN −1 − IN      ℏ d2 φN    IN − (Iℓ + Ibias )
                                                                               −            +         2
```

## Eq. (A49) — PDF page 14

```text
of Fig. 1 gives                                                                                          2
                                                                          −                −         eN d φN − (Iℓ + Ibias ) = 0.
                                                                                             
                                                                        −CN IN −1 + 1 + CN     IN + C
                    IN − (Iℓ + Ibias ) = q̇N              (A49)                                          dt2
                         IN −1 − IN = qN˙−1               (A50)
                        2
            q̇N −1   ℏ d φN      q̇N
```

## Eq. (A50) — PDF page 14

```text
                                                                          −                −         eN d φN − (Iℓ + Ibias ) = 0.
                                                                                             
                                                                        −CN IN −1 + 1 + CN     IN + C
                    IN − (Iℓ + Ibias ) = q̇N              (A49)                                          dt2
                         IN −1 − IN = qN˙−1               (A50)
                        2
            q̇N −1   ℏ d φN      q̇N
          −        +          +        = 0                (A51)         that can be recast as
```

## Eq. (A51) — PDF page 14

```text
                    IN − (Iℓ + Ibias ) = q̇N              (A49)                                          dt2
                         IN −1 − IN = qN˙−1               (A50)
                        2
            q̇N −1   ℏ d φN      q̇N
          −        +          +        = 0                (A51)         that can be recast as
            CN −1    2e dt2      CN

                           "                                                                         #
```

## Eq. (A52) — PDF page 14

```text
                           "                                                                         #
                                       2
                  −                  d   φ N −1       1   dφN −1                           1
                −CN          C
                             eJ,N −1            +                + Ic,N −1 sin φN −1 +         φ                               (A52)
                                         dt2       eJ,N −1 dt
                                                   R                                   e g,N −1 N −1
                                                                                       L
```

## Eq. (A53) — PDF page 14

```text
                                                                                       L
                           "                      !                                             #
                −
                                          CeN      d2 φN     1 dφN                       1
         + 1 + CN              CJ,N +                     +           + Ic,N sin φN +        φ    − (Iℓ + Ibias ) = 0          (A53)
                                                                                        e g,n N
                               e
                                                −
```

## Eq. (A54) — PDF page 14

```text
                 φm+1     m          m−1                         em + α         − (Iℓm + Ibias ) = 0.
                                         
                  n−1 + f n−1 + α   φ
                                 n−1 n−1   + 1 + Cn     α
                                                        e n φn + fn   e n φn                                                   (A54)

and collecting the terms appropriately, we obtain
   − +                  −           − m            − em          − −                  −
```

## Eq. (A55) — PDF page 15

```text
                aN,1 φm+1          m+1
                      N −1 + aN,2 φN
                                               m
                                       = bN,1 fN           em         m         m−1          m−1
                                                 −1 + bN,2 fN + bn,3 Iℓ + cN,1 φN −1 + cN,2 φN   + Ibias                       (A55)

where
                                            − +                         −
```

## Eq. (A56) — PDF page 15

```text
                                                                           +
                                  aN,1 = −CN αN −1         aN,2 = 1 + CN   α
                                                                           eN             aN,3 = 0
                                          −                               −
                                  bN,1 = CN                bN,2 = − 1 + CN                bN,3 = 1                            (A56)
                                          − −                             −    −
                                  cN,1 = CN αN −1          cN,2 = − 1 + CN   α
                                                                             eN            cn,3 = 0
```

## Eq. (A57) — PDF page 15

```text
                             a1,2     a1,3    0          ...       0               φm+1
                                                                                      1            A1
                             a2,1     a2,2    a2,3       ...       0               φm+1
                                                                                      2            A2
                              ..       ..      ..        ..        ..              ..      =       ..        ,                 (A57)
                               .          .       .         .       .               .               .
                              0        ...    aN −1,1 aN −1,2      aN −1,3         φm+1
                                                                                    N −1
```

## Eq. (A58) — PDF page 15

```text
                       An = bn,1 fn−1 + bn,2 fenm + bn,3 fn+1
                                                          m
                                                              + cn,1 φn−1 + cn,2 φm−1
                                                                                  n   + cn,3 φm−1
                                                                                                n+1                            (A58)
                                                        f or n = 1, ..., N − 1,     m = 1, 2, ...M,

                               A1 = b1,1 Iim + b1,2 fe1m + b1,3 f2m + c1,2 φm−1
```

## Eq. (A59) — PDF page 15

```text

                                    h                                                i
                     Iim+1 = Iim−1 + φm+1
                                      1   α1+ − Iim + f1m + φm−1
                                                              1  α1− + Ci V̇im − Ibias (2kωi )                                 (A59)
                                    "                                                 #
                                                 C N                    m−1 −               2k
                     Iℓm+1 = Iℓm−1 + fN
```

## Eq. (A60) — PDF page 15

```text
                                      m
                                        − 1+           Iℓm − φm+1
                                                               N
                                                                   +
                                                                  αN + φN    αN − Ibias          .                             (A60)
                                                  Cℓ                                      C N Rℓ
```

## Eq. (A61) — PDF page 15

```text
                                                                              x1,2 x1,3   0               ...      0
                                                                                                                         
                                                                             x2,1 x2,2 x2,3              ...      0      
                                                                              .    ..   ..               ..       ..     
                                                                              .       .    .                .            ,   (A61)
                                                                              .                                    .
  The coefficients ai,j , bi,j , and ci,j , with i, j ∈ [1, N ],
                                                                                                                          
```

## Eq. (A62) — PDF page 16

```text
   1      1 ℏ         en = Cn ℏ               Cn
      =              C                 Cn− =
  Ln
  e      Ln 2e                2e             Cn−1
                                                    (A62)
          ℏ              C    C
                           n−1 n
  C
```

