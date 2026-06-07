"""Regenerate all AutoFlow phantoms under ``data/``."""

from __future__ import annotations

import generate_phantom_P
import generate_phantoms


def main():
    selected = ["S", "U", "Y"]
    infos = generate_phantoms.generate_selected(selected)
    for name in selected:
        for line in generate_phantoms.summary_lines(name, infos[name]):
            print(line)

    truth = generate_phantom_P.generate_and_save()
    for line in generate_phantom_P.summary_lines(truth, generate_phantom_P.PHANTOM_H5_PATH):
        print(line)


if __name__ == "__main__":
    main()
