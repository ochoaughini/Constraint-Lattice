<?php
/**
 * Plugin Name: Constraint Lattice API Demo
 * Plugin URI:  https://github.com/ochoaughini/Constraint-Lattice
 * Description: Adds the shortcode [clattice_demo] to interact with the Constraint-Lattice Cloud Run API.
 * Version:     0.1.0
 * Author:      Constraint Lattice Team
 * License:     MIT
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit; // Abort if accessed directly.
}

/**
 * Renders the demo UI and JavaScript that calls the Cloud Run endpoint.
 *
 * Env vars expected in wp-config.php or server config:
 *   CLATTICE_API_URL  – e.g. https://constraint-lattice-xyz-uc.a.run.app
 *   CLATTICE_API_KEY  – tenant API key (or leave blank if endpoint is public)
 *
 * Usage in a post or page:
 *   [clattice_demo]
 *
 * @return string HTML block for front-end.
 */
function clattice_render_demo() {
    $api_url = getenv( 'CLATTICE_API_URL' ) ?: 'https://constraint-lattice-example.run.app';
    $api_key = getenv( 'CLATTICE_API_KEY' ); // optional

    ob_start();
    ?>
    <section id="clattice-demo" style="max-width:640px;margin:1rem auto;font-family:Arial,Helvetica,sans-serif;">
      <textarea id="cl-input" placeholder="Enter your prompt here …" style="width:100%;height:120px;margin-bottom:0.5rem;"></textarea>
      <select id="cl-constraint" style="margin-bottom:0.5rem;">
        <option value="json">Return JSON</option>
        <option value="formal">Formal Portuguese</option>
      </select>
      <button id="cl-submit" style="padding:0.4rem 1rem;">Apply Constraint</button>
      <pre id="cl-output" style="white-space:pre-wrap;background:#f6f8fa;padding:0.75rem;margin-top:1rem;min-height:4rem;border:1px solid #ddd;"></pre>
    </section>
    <script>
    (function () {
      const endpoint = <?php echo json_encode( rtrim( $api_url, '/' ) ); ?>;
      const apiKey   = <?php echo json_encode( $api_key ); ?>;
      document.getElementById('cl-submit').addEventListener('click', async () => {
        const prompt = document.getElementById('cl-input').value.trim();
        if (!prompt) { alert('Please enter a prompt.'); return; }
        const body = { prompt: prompt, output: prompt };
        const resp = await fetch(endpoint + '/govern', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(apiKey ? { 'X-API-Key': apiKey } : {})
          },
          body: JSON.stringify(body)
        });
        const data = await resp.json();
        document.getElementById('cl-output').textContent = data.moderated ?? JSON.stringify(data, null, 2);
      });
    })();
    </script>
    <?php
    return ob_get_clean();
}
add_shortcode( 'clattice_demo', 'clattice_render_demo' );
?>
