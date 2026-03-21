// SplitMix64 PRNG matching the original C implementation
// Uses BigInt for 64-bit arithmetic correctness

const MASK64 = 0xffffffffffffffffn;

export class SplitMix64 {
	private state: bigint;

	constructor(seed: number | bigint = 0) {
		this.state = BigInt(seed) & MASK64;
	}

	next(): bigint {
		this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
		let z = this.state;
		z = (((z ^ (z >> 30n)) & MASK64) * 0xbf58476d1ce4e5b9n) & MASK64;
		z = (((z ^ (z >> 27n)) & MASK64) * 0x94d049bb133111ebn) & MASK64;
		return (z ^ (z >> 31n)) & MASK64;
	}

	nextU32(): number {
		return Number(this.next() & 0xffffffffn);
	}

	nextU8(): number {
		return Number(this.next() & 0xffn);
	}

	// Returns a random number in [0, bound)
	nextBounded(bound: number): number {
		return this.nextU32() % bound;
	}
}
