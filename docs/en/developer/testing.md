# Testing

## Recommended Environment
Use:

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Supported Automated Suite

| Area | Main tests |
| --- | --- |
| smoke regression | `tests/test_smoke_phantoms.py` |
| relative-pressure phantom regression | `tests/test_pressure_gradient_phantom.py` |

## Expectations
- keep the automated suite limited to smoke and phantom regression coverage
- do not require a new targeted pytest file for every code change
- use manual verification and documentation updates when behavior changes outside the retained regression suite

## Useful Commands

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q
```
