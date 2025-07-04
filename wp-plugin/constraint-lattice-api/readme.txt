=== Constraint Lattice API Demo ===
Contributors: constraintlattice
Tags: ai, moderation, fastapi, llm
Requires at least: 5.9
Tested up to: 6.5
Stable tag: 0.1.0
License: MIT

A lightweight shortcode to demo Constraint-Lattice moderation directly inside WordPress pages.

== Description ==
This plugin ships a handy `[clattice_demo]` shortcode. Drop it into any page or post and visitors can submit a prompt which is sent to your Constraint-Lattice Cloud Run deployment. The response is rendered live.

Environment variables (set via wp-config.php or server):
* `CLATTICE_API_URL` – Fully-qualified base URL of your Cloud Run service.
* `CLATTICE_API_KEY` – Optional tenant API key (added in `X-API-Key`).

== Installation ==
1. Copy the `constraint-lattice-api` folder into `wp-content/plugins/`.
2. Activate *Constraint Lattice API Demo* in wp-admin → Plugins.
3. Define `CLATTICE_API_URL` in *wp-config.php*.
4. Optionally set `CLATTICE_API_KEY`.

== Usage ==
Insert the shortcode in Gutenberg or classic editor:
```
[clattice_demo]
```

== Frequently Asked Questions ==
*Does it store any data?* – No, calls go straight to your API. No logs are kept by the plugin itself.

*Is SSL required?* – Yes. Cloud Run forces HTTPS so your WordPress site must also serve over HTTPS to avoid Mixed-Content warnings.

== Changelog ==
= 0.1.0 =
* Initial release
