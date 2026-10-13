/** Display measurements describe composition, never fitness or ancestry. */
export type FieldMode = 'bytes' | 'identity' | 'confinement' | 'zeros' | 'diversity' | 'marker';

export interface CellRegion {
	x0: number;
	y0: number;
	x1: number;
	y1: number;
}

export interface Marker {
	position: number;
	value: number;
}

export interface RegionMeasurement {
	region: CellRegion;
	update: number;
	cellCount: number;
	uniqueTapes: number;
	diversity: number;
	topShare: number;
	topTape: number[];
	zeroFraction: number;
	markerMatches: number | null;
	markerFraction: number | null;
}

export function normalizeRegion(region: CellRegion, width: number, height: number): CellRegion {
	if (Object.values(region).some((value) => !Number.isInteger(value)))
		throw new Error('Region coordinates must be whole numbers.');
	const { x0, x1, y0, y1 } = region;
	if (Math.min(x0, x1, y0, y1) < 0 || Math.max(x0, x1) >= width || Math.max(y0, y1) >= height)
		throw new Error('Region coordinates are outside the lattice.');
	return { x0: Math.min(x0, x1), y0: Math.min(y0, y1), x1: Math.max(x0, x1), y1: Math.max(y0, y1) };
}

/** Explicit sampled CPU analysis. Input uses the GPU's word-padded cell layout. */
export function measureRegion(
	data: Uint8Array,
	width: number,
	height: number,
	length: number,
	selection: CellRegion,
	update: number,
	marker: Marker | null = null
): RegionMeasurement {
	const region = normalizeRegion(selection, width, height);
	const stride = Math.ceil(length / 4) * 4;
	if (data.length !== width * height * stride)
		throw new Error('Population sample does not match the lattice dimensions.');
	const counts = new Map<string, number>();
	let zeros = 0;
	let topCount = 0;
	let topKey = '';
	let matches = 0;
	for (let y = region.y0; y <= region.y1; y++) {
		for (let x = region.x0; x <= region.x1; x++) {
			const offset = (y * width + x) * stride;
			let key = '';
			for (let b = 0; b < length; b++) {
				const value = data[offset + b];
				key += String.fromCharCode(value);
				if (value === 0) zeros++;
			}
			const count = (counts.get(key) ?? 0) + 1;
			counts.set(key, count);
			if (count > topCount) {
				topCount = count;
				topKey = key;
			}
			if (marker && data[offset + marker.position] === marker.value) matches++;
		}
	}
	const cellCount = (region.x1 - region.x0 + 1) * (region.y1 - region.y0 + 1);
	return {
		region,
		update,
		cellCount,
		uniqueTapes: counts.size,
		diversity: counts.size / cellCount,
		topShare: topCount / cellCount,
		topTape: Array.from(topKey, (char) => char.charCodeAt(0)),
		zeroFraction: zeros / (cellCount * length),
		markerMatches: marker ? matches : null,
		markerFraction: marker ? matches / cellCount : null
	};
}
