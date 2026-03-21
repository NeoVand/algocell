// Simulation constants matching the original znah/zff implementation
export const SOUP_WIDTH = 200;
export const SOUP_HEIGHT = 200;
export const TAPE_LENGTH = 16;
export const PAIR_LENGTH = TAPE_LENGTH * 2; // 32 bytes
export const SOUP_SIZE = SOUP_WIDTH * SOUP_HEIGHT * TAPE_LENGTH; // 640,000 bytes
export const CELL_COUNT = SOUP_WIDTH * SOUP_HEIGHT; // 40,000 cells
export const MAX_BATCH_PAIR_N = 8192;
export const Z80_STEPS = 128;
export const TILE_SIZE = 4; // 4x4 pixel tiles for visualization
export const DEFAULT_SEED = 6;
export const DEFAULT_NOISE_EXP = 4; // noise = 1/2^4 = 1/16
