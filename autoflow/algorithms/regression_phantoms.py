"""The retained 24-input analytic matrix used by the regression suites."""
import json
from pathlib import Path
from .analytic_phantoms import RESOLUTIONS, SHAPES, create_analytic_phantom, save_analytic_phantom

REGRESSION_CASES = tuple(
    f'{spec.name}_{"x".join(f"{v:g}" for v in spacing)}mm'
    for spec in SHAPES for spacing in RESOLUTIONS
)

def generate_regression_matrix(directory, *, reuse=False):
    """Write six analytical configurations at four spacings, without external data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/'manifest.json'
    if reuse and manifest.is_file():
        saved = json.loads(manifest.read_text())
        if (saved.get('schema') == 'autoflow.regression_phantoms.v4'
                and [Path(row['file']).stem for row in saved.get('cases', [])] == list(REGRESSION_CASES)
                and all((directory/row['file']).is_file() for row in saved['cases'])):
            return saved['cases']
        raise ValueError('Existing analytic matrix is incomplete or outdated; use a new output directory')
    records = []
    for spec in SHAPES:
        for spacing in RESOLUTIONS:
            stem = f'{spec.name}_{"x".join(f"{v:g}" for v in spacing)}mm'
            data = create_analytic_phantom(spec, spacing)
            path = save_analytic_phantom(directory/(stem+'.h5'), data)
            records.append(dict(file=path.name, name=spec.name, spacing_mm=list(spacing),
                                shape=list(data['flow'].shape), truth_type=data['parameters']['truth_type'],
                                primary_reference=data['parameters']['primary_reference']))
    manifest.write_text(json.dumps(dict(schema='autoflow.regression_phantoms.v4',
                                       data_count=len(records), cases=records), indent=2))
    return records
