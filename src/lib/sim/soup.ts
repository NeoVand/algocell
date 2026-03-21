// CPU-side soup utilities for trace visualization and disassembly
// The main simulation runs on GPU; this handles hover/inspect on CPU

import { Z80_STEPS, getTapeLength, getPairLength, type GridType } from './constants';
import { Z80 } from './z80';

// Get a neighbor index for CPU-side trace (picks first neighbor: right for square, appropriate for hex)
function getNeighborIndex(cellIndex: number, width: number, height: number, gridType: GridType): number {
	const x = cellIndex % width;
	const y = Math.floor(cellIndex / width);

	if (gridType === 'hex') {
		const isOddRow = (y & 1) !== 0;
		// Pick the right neighbor for hex (same row, x+1)
		const nx = Math.min(x + 1, width - 1);
		return y * width + nx;
	}

	// Square: right neighbor
	const nx = Math.min(x + 1, width - 1);
	return y * width + nx;
}

// Run Z80 trace on a specific cell for visualization
// Returns a pairLength x Z80_STEPS RGBA image (reads=green, writes=red)
export function traceCell(soupData: Uint8Array, cellIndex: number, width: number = 200, height: number = 200, gridType: GridType = 'square'): Uint8Array {
	const tapeLength = getTapeLength(gridType);
	const pairLength = getPairLength(gridType);
	const wordsPerCell = Math.ceil(tapeLength / 4);
	const traceImage = new Uint8Array(pairLength * Z80_STEPS * 4);

	const neighborIndex = getNeighborIndex(cellIndex, width, height, gridType);

	// Read cell data from word-aligned soup buffer
	const pairMem = new Uint8Array(pairLength);
	const cellByteOffset = cellIndex * wordsPerCell * 4;
	const neighborByteOffset = neighborIndex * wordsPerCell * 4;
	for (let i = 0; i < tapeLength; i++) {
		pairMem[i] = soupData[cellByteOffset + i];
	}
	for (let i = 0; i < tapeLength; i++) {
		pairMem[tapeLength + i] = soupData[neighborByteOffset + i];
	}

	const z80 = new Z80();
	let currentStep = 0;

	z80.readByte = (addr: number) => {
		const maskedAddr = addr % pairLength;
		const pixelIdx = (currentStep * pairLength + maskedAddr) * 4;
		traceImage[pixelIdx + 1] = 255;
		traceImage[pixelIdx + 3] = 255;
		return pairMem[maskedAddr];
	};
	z80.writeByte = (addr: number, val: number) => {
		const maskedAddr = addr % pairLength;
		pairMem[maskedAddr] = val;
		const pixelIdx = (currentStep * pairLength + maskedAddr) * 4;
		traceImage[pixelIdx] = 255;
		traceImage[pixelIdx + 3] = 255;
	};

	for (currentStep = 0; currentStep < Z80_STEPS; currentStep++) {
		if (z80.halted) break;
		z80.step();
	}

	return traceImage;
}

export function getCellData(soupData: Uint8Array, cellIndex: number, gridType: GridType = 'square'): Uint8Array {
	const tapeLength = getTapeLength(gridType);
	const wordsPerCell = Math.ceil(tapeLength / 4);
	const byteOffset = cellIndex * wordsPerCell * 4;
	// Return only the valid tapeLength bytes (skip padding bytes in the last word)
	return soupData.slice(byteOffset, byteOffset + tapeLength);
}
