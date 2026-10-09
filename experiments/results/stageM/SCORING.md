# Stage M (the 8080 subset): pre-registered scoring

L = 16 (n = 20, horizon 300,000): first replicator load–push 20/20; t_rep median 650; closed successor 20/20; first replicator loop-bearing 0
  closers by mechanism: {('RET NZ', '-'): 20}
  first tapes: {'01 c5 01 c5': 13, '21 e5 21 e5': 6, '11 d5 11 d5': 1}
  finals (first 8 bytes) of closed worlds: {'ad e3 21 e3 21 c0 ad c0': 20}
L = 32 (n = 20, horizon 1,000,000): first replicator load–push 20/20; t_rep median 325; closed successor 0/20; first replicator loop-bearing 0
  closers by mechanism: {}
  first tapes: {'01 c5 01 c5': 14, '11 d5 11 d5': 4, '21 e5 21 e5': 2}
  finals (first 8 bytes) of closed worlds: {}

M1 (load–push first in ≥ 18/20 at both lengths): L16 20/20, L32 20/20 → MET
M2 (closed successor at L = 16 in ≥ 15/20 by return designs; kill < 10): 20/20 → MET
M3 (closure at L = 32 in ≤ 5/20): 0/20 → MET
