<script lang="ts">
	import type { LabMetrics } from '$lib/lab/types';
	let { history, current }: { history: LabMetrics[]; current: LabMetrics | null } = $props();
	const measures = [
		{
			key: 'zeroFraction',
			name: 'Zero bytes',
			color: '#83909A',
			explanation: 'Fraction of tape bytes equal to 00.'
		},
		{
			key: 'topShare',
			name: 'Top tape',
			color: '#168A8D',
			explanation: 'Cells carrying the most common exact tape.'
		},
		{
			key: 'diversity',
			name: 'Tape diversity',
			color: '#25313A',
			explanation: 'Distinct tapes divided by all cells.'
		}
	] as const;
	function path(key: (typeof measures)[number]['key']) {
		if (!history.length) return '';
		return history
			.map(
				(point, index) =>
					`${index ? 'L' : 'M'}${30 + (index / Math.max(history.length - 1, 1)) * 258},${90 - point[key] * 70}`
			)
			.join(' ');
	}
</script>

<section class="measurements" aria-labelledby="measurement-title">
	<div class="measurement-heading">
		<h2 id="measurement-title">Measurements</h2>
		<span>Sampled after each completed update</span>
	</div>
	<div class="plots">
		{#each measures as measure (measure.key)}
			<div class="plot">
				<div class="metric-title">
					<h3>{measure.name}</h3>
					<strong>{current ? `${(current[measure.key] * 100).toFixed(1)}%` : '—'}</strong>
				</div>
				<svg
					viewBox="0 0 300 112"
					role="img"
					aria-label={`${measure.name} over updates, from zero to one hundred percent. Current ${current ? (current[measure.key] * 100).toFixed(1) + ' percent' : 'unavailable'}`}
				>
					<path
						d="M30 20H288 M30 55H288 M30 90H288"
						stroke="#e6e8eb"
						stroke-width="1"
						fill="none"
					/>
					<text x="0" y="24">100</text><text x="13" y="94">0</text>
					{#if history.length > 1}<path
							d={path(measure.key)}
							stroke={measure.color}
							stroke-width="2.5"
							fill="none"
							stroke-linejoin="round"
						/>{:else}<text class="waiting" x="156" y="61" text-anchor="middle"
							>Step or run to record a trend</text
						>{/if}
					<text x="30" y="109">{history[0]?.update ?? 0}</text><text
						x="288"
						y="109"
						text-anchor="end">{current?.update ?? 0} updates</text
					>
				</svg>
				<p>{measure.explanation}</p>
			</div>
		{/each}
	</div>
	<p class="interpretation">
		A pattern can spread without transmitting variation. Use the culture and marker tests to ask
		what it does.
	</p>
</section>

<style>
	.measurements {
		padding: 20px 0 0;
	}
	.measurement-heading {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 12px;
	}
	h2 {
		font-size: 17px;
		margin: 0;
	}
	.measurement-heading span {
		font-size: 12px;
		color: var(--muted);
	}
	.plots {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 20px;
	}
	.plot {
		min-width: 0;
	}
	.metric-title {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 8px;
	}
	h3 {
		font-size: 13px;
		font-weight: normal;
		margin: 0;
	}
	strong {
		font-size: 22px;
		letter-spacing: -0.7px;
		font-variant-numeric: tabular-nums;
	}
	svg {
		width: 100%;
		display: block;
		margin-top: 5px;
	}
	text {
		font-family: Arial, sans-serif;
		font-size: 12px;
		fill: #63717b;
	}
	.waiting {
		font-size: 9px;
	}
	p {
		color: var(--muted);
		font-size: 12px;
		margin: 6px 0 0;
		line-height: 1.5;
	}
	.interpretation {
		margin-top: 16px;
		padding-top: 14px;
		border-top: 1px solid var(--line);
	}
	@media (max-width: 600px) {
		.plots {
			grid-template-columns: 1fr;
			gap: 20px;
		}
		.plot {
			display: grid;
			grid-template-columns: 100px minmax(0, 1fr);
			align-items: center;
			gap: 4px 10px;
		}
		.plot svg {
			grid-column: 2;
			grid-row: 1 / span 2;
		}
		.plot p {
			grid-column: 1;
			grid-row: 2;
		}
		.measurement-heading {
			display: block;
		}
		.measurement-heading span {
			display: block;
			margin-top: 4px;
		}
		strong {
			font-size: 19px;
		}
		.metric-title {
			display: block;
		}
		h3 {
			margin-bottom: 2px;
		}
	}
</style>
