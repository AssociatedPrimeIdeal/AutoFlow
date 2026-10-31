"""English voiceover drafts and picture timings for the silent videos."""
from pathlib import Path


def write_narration(compositor, output):
    opening='AutoFlow automatically analyzes four-dimensional flow data across different anatomical regions.'
    click=compositor.intro-.5
    sequence=[opening,
        'At the push of a button, it performs correction, segmentation, and topology reconstruction.',
        'Automatically placed analysis planes provide flow measurements throughout the cardiac cycle.',
        'A broad range of hemodynamic metrics characterizes blood-flow dynamics.',
        'Finally, an interactive quality-control report brings the results together, '
        'links flagged findings to the three-dimensional anatomy, and supports review of the complete analysis.',
    ]
    simultaneous=[opening,
        'At the push of a button, it performs correction, segmentation, topology reconstruction, and automatic plane placement.',
        'The outputs include time-resolved plane metrics, a broad range of hemodynamic measurements, '
        'and an interactive quality-control report.',
    ]
    blocks=[]
    for mode,title,sentences in [('sequence','Sequential workflow',sequence),
                            ('simultaneous','Simultaneous workflow',simultaneous)]:
        duration=compositor.final_start[mode]+compositor.closing
        text=' '.join(sentences)
        blocks.extend([f'AutoFlow - {title}',f'Video duration: {duration:.1f} seconds',
                       f'Word count: {len(text.split())}', '',text,'',
                       'Picture timings (the video has no recorded narration):',
                       '01.7 s  Input loading complete',f'{click:04.1f} s  Run All'])
        if mode=='sequence':
            for at,label in zip(compositor.starts,['Correction','Segmentation','Topology',
                                                    'Plane Metrics','Hemodynamics','QC Report']):
                blocks.append(f'{at:04.1f} s  {label}')
        else:blocks.append(f'{compositor.intro:04.1f} s  All six workflow panels appear')
        blocks.extend(['',''])
    path=Path(output)/'AutoFlow_voiceover_scripts.txt'
    path.write_text('\n'.join(blocks),encoding='utf-8')
    return path
