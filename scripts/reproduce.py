"""Plot archived Lorenz output or rerun its original code without a notebook UI.

The notebook remains the algorithm source. Archived and fresh results are kept
separate: rerunning cannot overwrite the evidence used in the README.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "LorenzExampleSINDy (2).ipynb"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_output(output):
    """Read only the main sweep, not the later diagnostic tables."""
    sweep = output.split("Deg   Noise", 1)[1].split("Seed analysis:", 1)[0]
    rows = []
    for line in sweep.splitlines():
        match = re.fullmatch(
            r"\s*(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.eE+-]+)\s+(true|false)\s*", line
        )
        if match:
            degree, noise, terms, error, support = match.groups()
            rows.append(dict(degree=int(degree), noise=float(noise),
                             library_size=int(terms), coeff_error=float(error),
                             support_correct=support == "true"))
    if len(rows) != 25 or len({(r['degree'], r['noise']) for r in rows}) != 25:
        raise ValueError("Expected the original 5-by-5 sweep in notebook output")
    return rows


def save_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path, provenance):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    import numpy as np

    degrees = sorted({r['degree'] for r in rows})
    noises = sorted({r['noise'] for r in rows})
    lookup = {(r['degree'], r['noise']): r for r in rows}
    support = np.array([[lookup[d, n]['support_correct'] for n in noises] for d in degrees])
    error = np.array([[lookup[d, n]['coeff_error'] for n in noises] for d in degrees])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axes[0].imshow(support, cmap=ListedColormap(['#edf0f4', '#007f73']), vmin=0, vmax=1)
    image = axes[1].imshow(np.maximum(error, 5e-7), cmap='magma_r',
                           norm=LogNorm(vmin=5e-7, vmax=2))
    for i, degree in enumerate(degrees):
        for j, noise in enumerate(noises):
            axes[0].text(j, i, 'YES' if support[i, j] else 'NO', ha='center', va='center',
                         color='white' if support[i, j] else '#334155', fontsize=10)
            value = error[i, j]
            label = '<5e-7' if value == 0 else f'{value:.2g}'
            axes[1].text(j, i, label, ha='center', va='center',
                         color='white' if value > 0.005 else '#172033', fontsize=9)
    for ax, title in zip(axes, ['Exact support recovery', 'Relative coefficient error']):
        ax.set(title=title, xlabel='Measurement noise standard deviation', ylabel='Polynomial degree',
               xticks=range(len(noises)), xticklabels=[f'{n:g}' for n in noises],
               yticks=range(len(degrees)), yticklabels=degrees)
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes[1], shrink=0.8, label='Frobenius error ratio (log scale)')
    fig.suptitle('Lorenz: quadratic terms suffice in the seed-42 sweep', fontsize=15, fontweight='bold')
    fig.supxlabel(provenance + '\nDegree 1 omits true quadratic terms; its error uses a truncated reference. '
                  'Archived zeros are rounded to six decimals.', fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plot', 'run'])
    args = parser.parse_args()
    notebook = json.loads(NOTEBOOK.read_text())
    if args.action == 'plot':
        output = ''.join(''.join(o.get('text', [])) for cell in notebook['cells']
                         for o in cell.get('outputs', []) if o['output_type'] == 'stream')
        rows = parse_output(output)
        folder = ROOT / 'results' / 'archived'
        folder.mkdir(parents=True, exist_ok=True)
        save_csv(rows, folder / 'lorenz_sweep.csv')
        plot(rows, ROOT / 'assets' / 'lorenz_sweep.png', 'Source: saved notebook output; one trial per cell, seed 42.')
        print('Saved results/archived/lorenz_sweep.csv and assets/lorenz_sweep.png')
        return

    # Skip the notebook's unpinned %pip cell; use requirements.txt instead.
    source = '\n\n'.join(''.join(c['source']) for c in notebook['cells']
                         if c['cell_type'] == 'code' and not ''.join(c['source']).lstrip().startswith('%'))
    folder = ROOT / 'results' / 'reproduced'
    folder.mkdir(parents=True, exist_ok=True)
    namespace = {'__name__': '__main__'}
    start = time.perf_counter()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
    elapsed = time.perf_counter() - start
    (folder / 'lorenz.txt').write_text(captured.getvalue())
    rows = namespace['results']
    save_csv(rows, folder / 'lorenz_sweep.csv')
    plot(rows, folder / 'lorenz_sweep.png', 'Source: fresh execution of original notebook code; seed 42.')
    import numpy as np
    import scipy
    import pysindy
    # A numerical diagnostic added for reproduction, not a historical validation criterion.
    custom = namespace['Xi2']
    reference = namespace['model'].coefficients().T
    if custom.shape != reference.shape:
        raise ValueError('Custom and PySINDy coefficient matrices have different shapes')
    assert namespace['check_support'](namespace['get_true_coefficients'](namespace['labels2']), custom)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__, pysindy=pysindy.__version__,
                    elapsed_seconds=elapsed,
                    notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                    pysindy_alpha=namespace['model'].optimizer.alpha,
                    pysindy_max_iter=namespace['model'].optimizer.max_iter,
                    pysindy_normalize_columns=namespace['model'].optimizer.normalize_columns,
                    lorenz_pysindy_support_equal=namespace['check_support'](custom, reference),
                    lorenz_pysindy_max_abs_difference=float(np.max(np.abs(custom - reference))))
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Completed original Lorenz experiment in {elapsed:.2f} s. Results: {folder.relative_to(ROOT)}')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
