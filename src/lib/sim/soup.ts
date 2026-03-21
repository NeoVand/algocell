// CPU-side soup utilities for trace visualization and disassembly
// The main simulation runs on GPU; this handles hover/inspect on CPU

import { SOUP_WIDTH, TAPE_LENGTH, PAIR_LENGTH, Z80_STEPS } from './constants';
import { Z80 } from './z80';

// Run Z80 trace on a specific cell for visualization
// Returns a PAIR_LENGTH x Z80_STEPS RGBA image (reads=green, writes=red)
export function traceCell(soupData: Uint8Array, cellIndex: number): Uint8Array {
	const traceImage = new Uint8Array(PAIR_LENGTH * Z80_STEPS * 4);
	const cellX = cellIndex % SOUP_WIDTH;
	const cellY = Math.floor(cellIndex / SOUP_WIDTH);

	const neighborX = Math.min(cellX + 1, SOUP_WIDTH - 1);
	const neighborIndex = cellY * SOUP_WIDTH + neighborX;

	const pairMem = new Uint8Array(PAIR_LENGTH);
	pairMem.set(
		soupData.subarray(cellIndex * TAPE_LENGTH, cellIndex * TAPE_LENGTH + TAPE_LENGTH),
		0
	);
	pairMem.set(
		soupData.subarray(neighborIndex * TAPE_LENGTH, neighborIndex * TAPE_LENGTH + TAPE_LENGTH),
		TAPE_LENGTH
	);

	const z80 = new Z80();
	let currentStep = 0;

	z80.readByte = (addr: number) => {
		const maskedAddr = addr & (PAIR_LENGTH - 1);
		const pixelIdx = (currentStep * PAIR_LENGTH + maskedAddr) * 4;
		traceImage[pixelIdx + 1] = 255;
		traceImage[pixelIdx + 3] = 255;
		return pairMem[maskedAddr];
	};
	z80.writeByte = (addr: number, val: number) => {
		const maskedAddr = addr & (PAIR_LENGTH - 1);
		pairMem[maskedAddr] = val;
		const pixelIdx = (currentStep * PAIR_LENGTH + maskedAddr) * 4;
		traceImage[pixelIdx] = 255;
		traceImage[pixelIdx + 3] = 255;
	};

	for (currentStep = 0; currentStep < Z80_STEPS; currentStep++) {
		if (z80.halted) break;
		z80.step();
	}

	return traceImage;
}

export function getCellData(soupData: Uint8Array, cellIndex: number): Uint8Array {
	const start = cellIndex * TAPE_LENGTH;
	return soupData.slice(start, start + TAPE_LENGTH);
}
