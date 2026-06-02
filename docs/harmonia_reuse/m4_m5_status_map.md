# Harmonia M4/M5 Status Map
| component | topology smoke | linear JC smoke | campaign | dataset | objective | nonlinear | status | blocker |
|---|---|---|---|---|---|---|---|---|
| JTL | PASS | PASS | PASS | PASS | PASS | not claimed | ready | none |
| RF-JTL / RF-SQUID | PASS | PASS | PASS | PASS | PASS | not claimed | linear ready | flux-bias semantics not validated |
| coupler | PASS | blocked | n/a | n/a | n/a | n/a | topology only | CircuitIR adapter lacks mutual-inductor K export |
| IPM | PASS | blocked | n/a | n/a | n/a | n/a | topology only | composed coupler K export missing |
| tiny nonlinear HB | PASS | one-port linearized S11 | PASS | n/a | n/a | PASS | tiny smoke ready | no residual exposed by current result API |
| old-Harmonia JTWPA standard experiment | staged | pending | pending | pending | pending | pending | staged | promotion validation pending |
| old-Harmonia RF-SQUID experiment | runnable tiny topology | PASS via RF-JTL | PASS | PASS | PASS | pending | staged runnable | flux-bias semantics pending |
| old-Harmonia IPM experiment | staged | blocked | pending | pending | pending | pending | staged | coupler K export missing |
| old-Harmonia directional coupler experiment | staged | blocked | pending | pending | pending | pending | staged | coupler K export missing |
