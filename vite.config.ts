import devtoolsJson from 'vite-plugin-devtools-json';
import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit(), devtoolsJson()],
	// The headless experiment runner writes result files under experiments/;
	// without this every batch download forces a full dev-server reload.
	server: { watch: { ignored: ['**/experiments/**'] } }
});
