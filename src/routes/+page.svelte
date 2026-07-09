<script lang="ts">
	// Landing / hub — routes to the experiments. The flagship soup experiment now
	// lives at /soup; new experiments (morphogenesis, gradient-based development)
	// are added here as they mature.
	import { resolve } from '$app/paths';

	interface Experiment {
		title: string;
		blurb: string;
		href: string;
		external?: boolean;
		glyph: string;
		accent: string;
		status?: string;
	}

	const experiments: Experiment[] = [
		{
			title: 'Computational Life',
			blurb: 'A soup of random bytes collides, copies, and evolves into self-replicating Z80 machine code — open-ended emergence from noise.',
			href: resolve('/soup'),
			glyph: '🧬',
			accent: '#3b82f6'
		},
		{
			title: 'Morphogenesis',
			blurb: 'A cellular automaton grows from a single seed and evolves toward a target shape — spatially invariant, and self-repairing.',
			href: resolve('/dev/morph/evolve'),
			glyph: '🌱',
			accent: '#10b981'
		},
		{
			title: 'Developmental Computation',
			blurb: 'A logic gate grown from a single seed cell — it computes, and when you damage it, regrows and still computes. Learned by gradient through development, running live on your GPU.',
			href: resolve('/devcomp'),
			glyph: '∇',
			accent: '#2dd4bf',
			status: 'live'
		}
	];
</script>

<svelte:head>
	<title>Algocell — experiments in artificial life</title>
</svelte:head>

<main class="wrap">
	<header class="hero">
		<h1>Algocell</h1>
		<p class="tag">Experiments in artificial life &amp; self-organizing computation — on a zillion tiny Z80 computers.</p>
	</header>

	<section class="grid" aria-label="Experiments">
		{#each experiments as e (e.title)}
			<a
				class="card"
				href={e.href}
				style="--accent:{e.accent}"
				target={e.external ? '_blank' : undefined}
				rel={e.external ? 'noopener noreferrer' : undefined}
			>
				<div class="top">
					<span class="glyph" aria-hidden="true">{e.glyph}</span>
					{#if e.status}<span class="badge">{e.status}</span>{/if}
				</div>
				<h2>{e.title}</h2>
				<p>{e.blurb}</p>
				<span class="go">{e.external ? 'Read the method →' : 'Open →'}</span>
			</a>
		{/each}
	</section>

	<footer class="foot">
		<span>Built on <a href="https://github.com/NeoVand/zilion" target="_blank" rel="noopener noreferrer">Zilion</a> — thousands of Z80 CPUs in parallel on the GPU.</span>
		<a href="https://github.com/NeoVand/algocell" target="_blank" rel="noopener noreferrer">Source</a>
	</footer>
</main>

<style>
	.wrap {
		min-height: 100vh;
		box-sizing: border-box;
		padding: clamp(24px, 6vw, 72px) clamp(20px, 5vw, 48px);
		display: flex;
		flex-direction: column;
		gap: clamp(28px, 5vw, 56px);
		background:
			radial-gradient(1200px 600px at 50% -10%, rgba(45, 212, 191, 0.08), transparent 60%),
			#0a0d12;
		color: #e6edf3;
		font-family:
			ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
	}
	.hero {
		max-width: 780px;
	}
	h1 {
		margin: 0;
		font-size: clamp(32px, 7vw, 56px);
		font-weight: 700;
		letter-spacing: -0.02em;
		background: linear-gradient(120deg, #e6edf3, #7dd3c8);
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
	}
	.tag {
		margin: 12px 0 0;
		font-size: clamp(14px, 2.4vw, 18px);
		line-height: 1.5;
		color: #9aa7b4;
		max-width: 62ch;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
		gap: 18px;
		max-width: 1080px;
		width: 100%;
	}
	.card {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 22px;
		border-radius: 16px;
		text-decoration: none;
		color: inherit;
		background: rgba(255, 255, 255, 0.025);
		border: 1px solid rgba(255, 255, 255, 0.07);
		transition:
			transform 0.16s ease,
			border-color 0.16s ease,
			background 0.16s ease;
		position: relative;
		overflow: hidden;
	}
	.card::before {
		content: '';
		position: absolute;
		inset: 0 0 auto 0;
		height: 3px;
		background: var(--accent);
		opacity: 0.7;
	}
	.card:hover {
		transform: translateY(-3px);
		border-color: color-mix(in srgb, var(--accent) 55%, transparent);
		background: color-mix(in srgb, var(--accent) 7%, rgba(255, 255, 255, 0.02));
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.glyph {
		font-size: 26px;
		line-height: 1;
		color: var(--accent);
	}
	.badge {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		padding: 3px 8px;
		border-radius: 999px;
		color: var(--accent);
		border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
	}
	.card h2 {
		margin: 4px 0 0;
		font-size: 18px;
		font-weight: 600;
	}
	.card p {
		margin: 0;
		font-size: 13.5px;
		line-height: 1.5;
		color: #9aa7b4;
		flex: 1;
	}
	.go {
		margin-top: 4px;
		font-size: 13px;
		font-weight: 600;
		color: var(--accent);
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 18px;
		justify-content: space-between;
		align-items: center;
		max-width: 1080px;
		width: 100%;
		margin-top: auto;
		padding-top: 20px;
		border-top: 1px solid rgba(255, 255, 255, 0.06);
		font-size: 12.5px;
		color: #6b7785;
	}
	.foot a {
		color: #9aa7b4;
	}
	.foot a:hover {
		color: #e6edf3;
	}
</style>
