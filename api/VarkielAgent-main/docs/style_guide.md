# Varkiel Agent Style Guide

## Documentation Standards
- Module docstrings: Purpose, key components, performance notes
- Class docstrings: Attributes, methods overview
- Function docstrings: Args, Returns, Raises

## Code Formatting
- PEP8 compliance (max line length: 120)
- Import order: standard lib, third party, local
- Type hints for all function signatures

## Error Handling
- Use logging for operational messages
- WARNING for recoverable errors
- ERROR for unrecoverable failures
- Include context in exceptions

## Performance Practices
- Vectorize operations
- Minimize memory copies
- Use caching where appropriate
