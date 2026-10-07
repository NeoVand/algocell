"""SplitMix64, bit-identical to src/lib/sim/prng.ts (seeds the soup and the per-step batch seeds)."""

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int = 0) -> None:
        self.state = int(seed) & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def next_u32(self) -> int:
        return self.next() & 0xFFFFFFFF
