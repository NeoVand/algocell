<script lang="ts">
	import { onMount } from 'svelte';

	let { chart }: { chart: string } = $props();
	let container: HTMLDivElement;
	let id = `mermaid-${Math.random().toString(36).slice(2, 8)}`;

	onMount(async () => {
		const mermaid = (await import('mermaid')).default;
		mermaid.initialize({
			startOnLoad: false,
			theme: 'dark',
			themeVariables: {
				darkMode: true,
				background: 'transparent',
				primaryColor: 'rgba(200,135,90,0.15)',
				primaryBorderColor: '#c8875a',
				primaryTextColor: '#e0cfc0',
				secondaryColor: 'rgba(200,135,90,0.08)',
				secondaryBorderColor: '#c8875a',
				secondaryTextColor: '#d0c0b0',
				tertiaryColor: 'rgba(200,135,90,0.06)',
				tertiaryBorderColor: 'rgba(200,135,90,0.3)',
				tertiaryTextColor: '#d0c0b0',
				lineColor: '#9a7050',
				textColor: '#d0c0b0',
				mainBkg: 'rgba(200,135,90,0.12)',
				nodeBorder: '#c8875a',
				nodeTextColor: '#e0cfc0',
				clusterBkg: 'rgba(255,255,255,0.03)',
				clusterBorder: 'rgba(255,255,255,0.08)',
				edgeLabelBackground: 'transparent',
				noteBkgColor: 'rgba(200,135,90,0.1)',
				noteBorderColor: '#c8875a',
				noteTextColor: '#d0c0b0',
				fontSize: '11px',
				fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif'
			},
			flowchart: {
				htmlLabels: true,
				curve: 'basis',
				padding: 10,
				nodeSpacing: 30,
				rankSpacing: 35,
				defaultRenderer: 'dagre-wrapper'
			}
		});
		const { svg } = await mermaid.render(id, chart.trim());
		container.innerHTML = svg; // eslint-disable-line svelte/no-dom-manipulating -- Mermaid requires direct DOM injection
	});
</script>

<div bind:this={container} class="mermaid-container"></div>

<style>
	.mermaid-container {
		width: 100%;
		margin: 8px 0;
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.2);
		border: 1px solid rgba(255, 255, 255, 0.04);
		padding: 12px 8px;
		overflow-x: auto;
		display: flex;
		justify-content: center;
	}
	.mermaid-container :global(svg) {
		max-width: 100%;
		height: auto;
	}
	/* Round all node rects */
	.mermaid-container :global(.node rect),
	.mermaid-container :global(.node polygon) {
		rx: 8;
		ry: 8;
	}
	/* Style edge labels */
	.mermaid-container :global(.edgeLabel) {
		font-size: 10px !important;
		background: transparent !important;
	}
	.mermaid-container :global(.edgeLabel rect) {
		fill: rgba(30, 22, 18, 0.85) !important;
		stroke: none !important;
		rx: 4;
		ry: 4;
	}
</style>
