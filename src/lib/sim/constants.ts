// Simulation constants matching the original znah/zff implementation

// Grid topology
export type GridType = 'square' | 'hex';

export interface GridConfig {
	width: number;
	height: number;
	gridType: GridType;
}

// Default grid dimensions
export const SOUP_WIDTH = 200;
export const SOUP_HEIGHT = 200;

// Per-cell constants
export const TAPE_LENGTH = 16; // square mode: 4×4 = 16 bytes per cell
export const PAIR_LENGTH = TAPE_LENGTH * 2; // 32 bytes

// Hex mode: 19 bytes per cell (3-4-5-4-3 hexagonal arrangement)
export const HEX_TAPE_LENGTH = 19;
export const HEX_PAIR_LENGTH = HEX_TAPE_LENGTH * 2; // 38 bytes

export function getTapeLength(gridType: GridType): number {
	return gridType === 'hex' ? HEX_TAPE_LENGTH : TAPE_LENGTH;
}

export function getPairLength(gridType: GridType): number {
	return gridType === 'hex' ? HEX_PAIR_LENGTH : PAIR_LENGTH;
}

// Derived defaults (for backward compatibility)
export const SOUP_SIZE = SOUP_WIDTH * SOUP_HEIGHT * TAPE_LENGTH; // 640,000 bytes
export const CELL_COUNT = SOUP_WIDTH * SOUP_HEIGHT; // 40,000 cells

// Batch / execution
export const MAX_BATCH_PAIR_N = 8192;
export const Z80_STEPS = 128;
export const TILE_SIZE = 4; // 4x4 pixel tiles for visualization

// Defaults
export const DEFAULT_SEED = 6;
export const DEFAULT_NOISE_EXP = 4; // noise = 1/2^4 = 1/16
export const DEFAULT_GRID_CONFIG: GridConfig = {
	width: SOUP_WIDTH,
	height: SOUP_HEIGHT,
	gridType: 'square'
};

// Hex geometry
export const HEX_HEIGHT_RATIO = 0.8660254037844386; // sqrt(3)/2
