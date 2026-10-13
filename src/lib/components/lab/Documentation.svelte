<script lang="ts">
	import { ABOUT_SECTIONS, GUIDE_SECTIONS } from '$lib/lab/content';
	let { kind, onclose }: { kind: 'about' | 'guide'; onclose: () => void } = $props();
	let sections = $derived(kind === 'about' ? ABOUT_SECTIONS : GUIDE_SECTIONS);
	function openDialog(dialog: HTMLDialogElement) {
		const invoker = document.activeElement instanceof HTMLElement ? document.activeElement : null;
		dialog.showModal();
		return () => {
			dialog.close();
			invoker?.focus();
		};
	}
</script>

<dialog {@attach openDialog} {onclose} aria-labelledby="document-title">
	<header>
		<div>
			<span class="document-brand">Algocell</span>
			<h2 id="document-title">
				{kind === 'about' ? 'Copying is not yet inheriting.' : 'A guide to the instrument'}
			</h2>
		</div>
		<button class="icon-button" onclick={onclose} aria-label={`Close ${kind}`}>×</button>
	</header>
	<div class="document-body">
		<nav aria-label="On this page">
			{#each sections as section (section.id)}<a href={`#${section.id}`}>{section.title}</a>{/each}
		</nav>
		<article>
			{#each sections as section (section.id)}<section id={section.id}>
					<h3>{section.title}</h3>
					{#each section.paragraphs as paragraph (paragraph)}<p>
							{paragraph}
						</p>{/each}{#if section.steps}<ol>
							{#each section.steps as step (step)}<li>{step}</li>{/each}
						</ol>{/if}{#if section.links}<div class="reference-links">
							<!-- eslint-disable svelte/no-navigation-without-resolve -- These curated references are absolute external URLs. -->
							{#each section.links as link (link.href)}<a
									href={link.href}
									target="_blank"
									rel="noreferrer">{link.label}</a
								>{/each}
							<!-- eslint-enable svelte/no-navigation-without-resolve -->
						</div>{/if}
				</section>{/each}
		</article>
	</div>
</dialog>

<style>
	dialog {
		margin: auto;
		border: 1px solid var(--line);
		border-radius: 12px;
		padding: 0;
		width: min(900px, calc(100vw - 48px));
		max-height: calc(100dvh - 48px);
		color: var(--ink);
		background: #fff;
		box-shadow: 0 24px 100px #25313a30;
	}
	dialog::backdrop {
		background: #25313a85;
	}
	header {
		padding: 28px 32px;
		border-bottom: 1px solid var(--line);
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 20px;
	}
	.document-brand {
		font-size: 13px;
		color: var(--teal);
		font-weight: bold;
	}
	h2 {
		font-size: 28px;
		letter-spacing: -0.7px;
		line-height: 1.2;
		margin: 8px 0 0;
		max-width: 24ch;
	}
	.icon-button {
		font-size: 25px;
		flex-shrink: 0;
	}
	.document-body {
		display: grid;
		grid-template-columns: 190px minmax(0, 1fr);
		gap: 36px;
		padding: 28px 32px 40px;
		overflow-y: auto;
		max-height: calc(100dvh - 220px);
	}
	nav {
		display: flex;
		flex-direction: column;
		gap: 14px;
		font-size: 12px;
		align-self: start;
	}
	nav a {
		color: var(--muted);
		text-decoration: none;
		line-height: 1.4;
	}
	nav a:hover {
		color: var(--teal);
	}
	article {
		min-width: 0;
	}
	article section {
		margin-bottom: 32px;
		scroll-margin: 20px;
	}
	h3 {
		font-size: 19px;
		line-height: 1.3;
		margin: 0 0 12px;
	}
	p,
	li {
		font-size: 15px;
		line-height: 1.7;
		color: #4c5a63;
	}
	p {
		margin: 0 0 14px;
	}
	ol {
		padding-left: 20px;
	}
	li {
		margin: 10px 0;
	}
	.reference-links {
		display: flex;
		gap: 10px 20px;
		flex-wrap: wrap;
		font-size: 13px;
	}
	@media (max-width: 650px) {
		dialog {
			width: calc(100vw - 20px);
			max-height: calc(100dvh - 20px);
		}
		header {
			padding: 22px;
		}
		h2 {
			font-size: 23px;
		}
		.document-body {
			display: block;
			padding: 22px;
			max-height: calc(100dvh - 175px);
		}
		nav {
			flex-direction: row;
			flex-wrap: wrap;
			gap: 10px 15px;
			padding-bottom: 24px;
			margin-bottom: 24px;
			border-bottom: 1px solid var(--line);
		}
		p,
		li {
			font-size: 14px;
		}
	}
</style>
