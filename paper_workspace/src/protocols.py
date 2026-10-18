"""Scenario labels transcribed from annotations.pdf Supplementary Tables 2a-d.
cobedding yes = twins sharing cot (only one monitored with radar).
position: prone / supine / side (7.2 ambiguous in scan -> supine, see note).
"""
MEASURE_INFO = {
    # m: (position, cobedding_twins, lid)
    '1.1': ('supine', True, 'half-opened'),
    '1.2': ('prone', True, 'half-opened'),
    '1.3': ('supine', True, 'half-opened'),
    '2.1': ('supine', True, 'half-opened'),
    '2.2': ('supine', True, 'half-opened'),
    '2.3': ('supine', True, 'half-opened'),
    '3.1': ('prone', False, 'closed'),
    '3.2': ('prone', True, 'closed'),
    '3.3': ('side', True, 'closed'),  # paper discards side (low SNR)
    '4.1': ('supine', False, 'closed'),
    '4.2': ('supine', False, 'closed'),
    '4.3': ('prone', False, 'closed'),
    '5.1': ('supine', False, 'closed'),
    '5.2': ('supine', False, 'closed'),
    '5.3': ('prone', False, 'closed'),
    '6.1': ('supine', False, 'closed'),
    '6.2': ('prone', False, 'closed'),
    '6.3': ('supine', False, 'closed'),
    '7.1': ('supine', False, 'closed'),
    # 7.2 scan faint: assumed supine twins (consistent with 7.1/7.3 pattern + twins count)
    '7.2': ('supine', True, 'closed'),
    '7.3': ('prone', True, 'closed'),
    '8.1': ('supine', False, 'closed'),
    '8.2': ('supine', True, 'closed'),
    '8.3': ('prone', True, 'closed'),
    '9.1': ('supine', False, 'closed'),
    '9.2': ('supine', False, 'closed'),
    '9.3': ('supine', False, 'closed'),
    '10.1': ('supine', False, 'closed'),
    '10.2': ('prone', False, 'closed'),
    '10.3': ('prone', False, 'closed'),  # no radar file
    '11.1': ('supine', False, 'closed'),
    '11.2': ('supine', False, 'closed'),
    '11.3': ('supine', False, 'closed'),
    '12.1': ('prone', False, 'closed'),
    '12.2': ('prone', False, 'closed'),
    '12.3': ('supine', False, 'closed'),
}

# Paper valid subset: 18 measurements after excluding recording issues (1.2/3.2/10.3
# missing files), side position (3.3), and heavy ref-saturation. Supine 11 / prone 7,
# single 14 / twins 4 within those 18. Exact 18 list is not named in the paper;
# use valid_paired - side as candidate pool (32), then ref-valid-fraction gate.
