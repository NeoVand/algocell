// Checkpoint format: valid PNG + appended binary payload after IEND chunk
// Opening in Preview/image viewer → shows simulation screenshot
// Loading in the app → detects ACEL magic, extracts full state

const MAGIC = new Uint8Array([0x41, 0x43, 0x45, 0x4c]); // "ACEL"

export interface CheckpointMetadata {
	version: 1;
	gridWidth: number;
	gridHeight: number;
	gridType: 'square' | 'hex';
	seed: number;
	noiseExp: number;
	z80Steps: number;
	pairCount: number;
	suppressPatterns: string[];
	batchCount: number;
	timestamp: number;
}

export async function saveCheckpoint(
	pngBlob: Blob,
	metadata: CheckpointMetadata,
	soupData: Uint8Array
): Promise<Blob> {
	const pngBytes = new Uint8Array(await pngBlob.arrayBuffer());
	const metaJson = new TextEncoder().encode(JSON.stringify(metadata));

	// Gzip metadata and soup
	const metaGzipped = await gzipCompress(metaJson);
	const soupGzipped = await gzipCompress(soupData);

	// Payload: MAGIC(4) + metaLen(4) + soupLen(4) + metaGzipped + soupGzipped
	const payloadSize = 4 + 4 + 4 + metaGzipped.length + soupGzipped.length;
	const payload = new Uint8Array(payloadSize);
	const dv = new DataView(payload.buffer);

	payload.set(MAGIC, 0);
	dv.setUint32(4, metaGzipped.length, true);
	dv.setUint32(8, soupGzipped.length, true);
	payload.set(metaGzipped, 12);
	payload.set(soupGzipped, 12 + metaGzipped.length);

	// Concatenate PNG + payload
	const result = new Uint8Array(pngBytes.length + payload.length);
	result.set(pngBytes);
	result.set(payload, pngBytes.length);

	return new Blob([result], { type: 'application/octet-stream' });
}

export async function loadCheckpoint(
	data: Uint8Array
): Promise<{ metadata: CheckpointMetadata; soupData: Uint8Array } | null> {
	// Find IEND chunk, then look for ACEL magic after it
	const iendPos = findIEND(data);
	if (iendPos < 0) return null;

	// IEND chunk: length(4) + "IEND"(4) + CRC(4) = 12 bytes
	const payloadStart = iendPos + 12;
	if (payloadStart + 12 > data.length) return null;

	// Check ACEL magic
	if (
		data[payloadStart] !== 0x41 ||
		data[payloadStart + 1] !== 0x43 ||
		data[payloadStart + 2] !== 0x45 ||
		data[payloadStart + 3] !== 0x4c
	) {
		return null;
	}

	const dv = new DataView(data.buffer, data.byteOffset + payloadStart);
	const metaLen = dv.getUint32(4, true);
	const soupLen = dv.getUint32(8, true);

	const metaGzipped = data.slice(payloadStart + 12, payloadStart + 12 + metaLen);
	const soupGzipped = data.slice(
		payloadStart + 12 + metaLen,
		payloadStart + 12 + metaLen + soupLen
	);

	const metaBytes = await gzipDecompress(metaGzipped);
	const metadata: CheckpointMetadata = JSON.parse(new TextDecoder().decode(metaBytes));

	const soupData = await gzipDecompress(soupGzipped);

	return { metadata, soupData };
}

async function gzipCompress(data: Uint8Array): Promise<Uint8Array> {
	const buf = new ArrayBuffer(data.byteLength);
	new Uint8Array(buf).set(data);
	const stream = new Blob([buf]).stream().pipeThrough(new CompressionStream('gzip'));
	return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function gzipDecompress(data: Uint8Array): Promise<Uint8Array> {
	const buf = new ArrayBuffer(data.byteLength);
	new Uint8Array(buf).set(data);
	const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
	return new Uint8Array(await new Response(stream).arrayBuffer());
}

function findIEND(data: Uint8Array): number {
	// Search backward for IEND chunk type bytes: 49 45 4E 44
	// The IEND chunk starts with its length field (always 0), so we look for
	// 00 00 00 00 49 45 4E 44
	for (let i = data.length - 8; i >= 8; i--) {
		if (
			data[i] === 0x00 &&
			data[i + 1] === 0x00 &&
			data[i + 2] === 0x00 &&
			data[i + 3] === 0x00 &&
			data[i + 4] === 0x49 &&
			data[i + 5] === 0x45 &&
			data[i + 6] === 0x4e &&
			data[i + 7] === 0x44
		) {
			return i;
		}
	}
	return -1;
}
