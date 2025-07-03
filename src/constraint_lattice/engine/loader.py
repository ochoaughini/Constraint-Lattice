import importlib
import logging
from typing import Any, List, Sequence
import yaml


def load_constraint_class(class_name: str, search_modules: list[str]) -> Any:
    for module_name in search_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f"Constraint class '{class_name}' not found in modules: {search_modules}"
    )


def load_constraints_from_yaml(
    yaml_path: str, profile: str, search_modules: list[str]
) -> list:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    raw_entries = config["profiles"][profile]
    constraints: list[Any] = []
    for entry in raw_entries:
        # --- Flexible YAML schema ----------------------------------------------------
        # A profile item can be either:
        #   1. A *string* specifying the constraint class name.
        #   2. A *mapping* with at minimum a ``class`` key and optional metadata
        #      such as ``engine: jax`` or per-constraint configuration.
        if isinstance(entry, str):
            class_name = entry
            engine_hint = None
        elif isinstance(entry, dict):
            class_name = entry.get("class")
            if class_name is None:
                raise ValueError("Constraint mapping must include a 'class' key.")
            engine_hint = entry.get("engine")  # Currently informational only
        else:
            raise TypeError("Profile entries must be string or mapping.")

        cls = load_constraint_class(class_name, search_modules)
        # ------------------------------------------------------------------
        # Optional JAX integration.  If the YAML entry specifies ``engine: jax``
        # *and* the runtime has JAX enabled, we transparently wrap the
        # constraint into a compiled JAX closure.  We try a couple of
        # heuristics so that users can either:
        #   • Provide a *callable* class (e.g. a plain function) – or –
        #   • Provide a normal class with an ``enforce_constraint`` method.
        # ------------------------------------------------------------------
        if engine_hint == "jax":
            try:
                from engine.jax_backend import JAXConstraint  # type: ignore
                if callable(cls):  # Function-style constraint
                    instance = JAXConstraint(cls)  # type: ignore[arg-type]
                else:
                    base = cls()
                    enforce = getattr(base, "enforce_constraint", None)
                    if callable(enforce):
                        try:
                            jax_pred = JAXConstraint(enforce)  # type: ignore[arg-type]

                            def _jax_enforce(self, *a, **kw):  # noqa: D401
                                # Delegates to the compiled JAX predicate stored in closure.
                                return jax_pred(*a, **kw)

                            # Bind the compiled version to the instance.
                            setattr(base, "enforce_constraint", _jax_enforce.__get__(base, base.__class__))
                            instance = base
                        except Exception as wrap_exc:  # pragma: no cover
                            logging.warning(
                                "Failed to wrap %s with JAX: %s; falling back to pure Python.",
                                class_name,
                                wrap_exc,
                            )
                            instance = base
                    else:
                        logging.warning(
                            "engine: jax specified for %s but no enforce_constraint found; using raw instance.",
                            class_name,
                        )
                        instance = base
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to enable JAX for %s: %s", class_name, exc)
                instance = cls()
        else:
            instance = cls()
        constraints.append(instance)
    return constraints
