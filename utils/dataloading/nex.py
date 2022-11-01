import nex.nexfile
import pandas as pd
import numpy as np
from collections import defaultdict
import os

def loadNex(fname):
    units, spikes, trials, markers, _ = readNex(fname)
    raw_ecodes = preprocessMarkers(markers)
    paradigm_ecode_table = getParadigmEcodeTable(raw_ecodes)
    ecodes = parseEcodes(raw_ecodes, paradigm_ecode_table)
    header, trials = getTrialHeaderTables(ecodes, trials)
    
    return header, trials, units, spikes

def readNex(fname):
    reader = nex.nexfile.Reader()
    fileData = reader.ReadNexFile(os.path.join(os.getcwd(), fname))


    neurons = [var for var in fileData['Variables'] if var['Header']['Type'] == 0]

    units_raw = [
        [var['Header']['Name'], var['Header']['Count'], var['Header']['Version']] 
        for var in neurons
    ]
    unit_idx = [var['Header']['Unit'] for var in neurons]
    units = pd.DataFrame(units_raw, pd.Index(unit_idx, name='Unit'), columns=['Name', 'Count', 'Version'])

    spikes = pd.DataFrame({
        'Unit' : np.concatenate([
            np.ones(var['Timestamps'].shape, dtype=int) * var['Header']['Unit']
            for var in neurons
        ]),
        'Timestamp' : np.concatenate([
            var['Timestamps']
            for var in neurons
        ])
    })

    events = [var for var in fileData['Variables'] if var['Header']['Type'] == 1]

    # Start and stop define epochs
    for event in events:
        if 'Start' in event['Header']['Name']:
            start = event
        elif 'Stop' in event['Header']['Name']:
            stop = event

    if start['Timestamps'].size == stop['Timestamps'].size +1:
        print('Warning: mismatched start and stop lengths. Removing trialing start.')
        start['Timestamps'] = start['Timestamps'][:-1]

    assert start['Timestamps'].size == stop['Timestamps'].size, f'Error: mismatched start and stop lengths'

    n_trials = stop['Header']['Count']
    trials = pd.DataFrame({
        'start_time' : start['Timestamps'],
        'stop_time' : stop['Timestamps']
    }, index=pd.Series(np.arange(n_trials), name='id'))

    intervals = [var for var in fileData['Variables'] if var['Header']['Type'] == 2]
    waveforms = [var for var in fileData['Variables'] if var['Header']['Type'] == 3]
    populations = [var for var in fileData['Variables'] if var['Header']['Type'] == 4]
    continuous_vars = [var for var in fileData['Variables'] if var['Header']['Type'] == 5]

    #ecodes
    markers = [var for var in fileData['Variables'] if var['Header']['Type'] == 6]

    return units, spikes, trials, markers, fileData

def preprocessMarkers(markers):
    ### LOAD ECODES TO DATAFRAME

    # The one marker is the ecodes
    # If these fail then there are multiple DIO ports and something is weird
    assert len(markers) == 1
    assert len(markers[0]['Fields']) == 1
    assert markers[0]['Fields'][0]['Name'] == 'DIO'


    ecodes = np.array(markers[0]['Fields'][0]['Markers'])
    ecodes_ts = markers[0]['Timestamps']

    # Rex 1000 level ecodes
    ecodes_rex = ecodes[np.logical_and(ecodes >= 1000, ecodes < 2000)]
    ecodes_rex_ts = ecodes_ts[np.logical_and(ecodes >= 1000, ecodes < 2000)]
    ecodes_rex_data = [np.nan] * ecodes_rex.size

    # Lab 7000 and 8000 ecodes
    ecodes_lab_raw = ecodes[ecodes >= 2000]
    ecodes_lab_raw_ts = ecodes_ts[ecodes >= 2000]
    n_ecodes_lab_raw = ecodes_lab_raw.size

    # Check which codes are bookended and verify each code has only one length
    ecodes_lab_nondata_idx = np.nonzero(np.logical_and(ecodes_lab_raw >= 7000, ecodes_lab_raw < 9000))[0]
    ecodes_lab_nondata = ecodes_lab_raw[ecodes_lab_nondata_idx]
    ecodes_lab_numdata = np.diff(np.append(ecodes_lab_nondata_idx, n_ecodes_lab_raw)) - 1
    ecode_lab_data_lens = defaultdict(set)
    for i in range(ecodes_lab_nondata.size):
        ecode_lab_data_lens[ecodes_lab_nondata[i]].add(ecodes_lab_numdata[i])

    bookended_ecodes = set()
    for code, val in ecode_lab_data_lens.items():
        if len(val) == 1:
            # The ecode has a single data length. This is good!
            continue
        elif len(val) == 2 and 0 in val:
            # The ecode has two values and one of them is zero, then it's bookended correctly :)
            # This isn't strictly true, but it's a close approximation that is unlikely to break
            bookended_ecodes.add(code)
        else:
            raise Exception(f'Ecode: {code} has invalid data lengths - {val}')

    # Construct lab ecode lists
    ecodes_lab = []
    ecodes_lab_ts = []
    ecodes_lab_data = []

    in_bookend = False
    for i in range(n_ecodes_lab_raw):
        ecode = ecodes_lab_raw[i]
        if ecode >= 7000 and ecode < 9000:
            if in_bookend:
                if bookended_ecode == ecode:
                    in_bookend = False
                    continue
                else:
                    raise Exception(f'Ecode error: bookend corruption. Got {ecode}. Expected {bookended_ecode}.')
            ecodes_lab.append(ecode)
            ecodes_lab_ts.append(ecodes_lab_raw_ts[i])
            ecodes_lab_data.append([])
        elif ecode >= 2000 and ecode < 7000:
            ecodes_lab_data[-1].append(ecode)
        else:
            raise Exception(f'Ecode error: unexpected code - {ecode}')
        
        if ecode in bookended_ecodes:
            in_bookend = True
            bookended_ecode = ecode


    # Construct pandas dataframe
    raw_ecodes = pd.DataFrame({'ecode' : np.concatenate([ecodes_rex, ecodes_lab]), 
                            'timestamp' : np.concatenate([ecodes_rex_ts, ecodes_lab_ts]), 
                            'data_raw' : ecodes_rex_data + ecodes_lab_data})

    raw_ecodes = raw_ecodes.sort_values(by='timestamp')
    raw_ecodes = raw_ecodes.reset_index(drop=True) 

    return raw_ecodes

def getParadigmEcodeTable(raw_ecodes):

    ### Parse ecode data using paradigm specific ecode table ###
    OFFSET = 4000
    parse_int = lambda arr : np.array(arr) - OFFSET

    # Retrieve the proper ecode table csv
    paradigm_code = 8999
    paradigm_ecodes = raw_ecodes[raw_ecodes.ecode == paradigm_code]
    assert paradigm_ecodes.shape[0] == 1, "CORRUPTION: Two paradigm ecodes this file"

    paradigm_id = parse_int(paradigm_ecodes.data_raw.iloc[0])[0]

    paradigm_directory = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecodes', 'ECODES_DIRECTORY.csv'))
    paradigm = paradigm_directory[paradigm_directory['paradigm id'] == paradigm_id]
    assert paradigm.shape[0] == 1, f"Paradigm with id ({paradigm_id}) not known!"
    paradigm_file = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecodes', paradigm['ecodes file'].iloc[0]))
    ecodes_table = pd.read_csv(paradigm_file)

    return ecodes_table

def parseEcodes(raw_ecodes, ecodes_table):
    # Parsing functions
    OFFSET = 4000
    parse_long = lambda arr : (np.array(arr) - OFFSET).astype(np.uint8).view(dtype=np.int32)
    parse_double = lambda arr : (np.array(arr) - OFFSET).astype(np.uint8).view(dtype=np.double)
    parse_int = lambda arr : np.array(arr) - OFFSET
    parse_time = lambda arr : arr
    parse_dict = {
        'time' : parse_time,
        'int' : parse_int,
        'long' : parse_long,
        'double' : parse_double
    }

    all_codes_data = raw_ecodes.data_raw.copy()
    for i in range(ecodes_table.shape[0]):
        code = ecodes_table.iloc[i,2]
        dtype = ecodes_table.iloc[i,3]
        code_slice = raw_ecodes.ecode == code
        all_codes_data[code_slice] = all_codes_data[code_slice].apply(parse_dict[dtype])

    clean_data = lambda arr : arr[0] if hasattr(arr, 'size') and arr.size == 1 else arr
    raw_ecodes['data'] = all_codes_data.apply(clean_data)

    ecodes = raw_ecodes.merge(ecodes_table, how='inner', on='ecode')

    return ecodes

def getTrialHeaderTables(ecodes, trials):
    # Check for orphaned ecodes
    trial_ecodes = ecodes[np.logical_or(ecodes.ecode.between(7000, 7999), ecodes.ecode.between(1000, 1999))]
    header_ecodes = ecodes[ecodes.ecode.between(8000, 8999)]

    n_trials = trials.shape[0]
    # codes between trials (orphaned)
    orphaned_ecodes = []
    orphaned_ecodes.append(ecodes[ecodes.timestamp < trials.iloc[0,0]])
    orphaned_ecodes.extend(
    [
        ecodes[ecodes.timestamp.between(trials.iloc[i,1], trials.iloc[i+1,0])]
        for i in range(n_trials-1)
    ])
    orphaned_ecodes.append(ecodes[ecodes.timestamp > trials.iloc[-1,1]])
    orphaned_ecodes = pd.concat(orphaned_ecodes)

    if orphaned_ecodes.shape[0] != 0:
        print(f'Warning: {orphaned_ecodes.shape[0]} orphaned ecodes!')

    # Extending trials df
    columns = trial_ecodes.column.unique()
    for c in columns:
        values = []
        for t_idx in range(n_trials):
            time_mask = trial_ecodes.timestamp.between(trials.iloc[t_idx,0], trials.iloc[t_idx,1])
            col_mask = trial_ecodes.column == c
            ecode = trial_ecodes[np.logical_and(time_mask, col_mask)]
            if ecode.shape[0] == 0:
                values.append(np.nan)
            elif ecode.shape[0] == 1:
                ecode = ecode.iloc[0]
                if ecode.type == 'time':
                    values.append(ecode.timestamp)
                else:
                    values.append(ecode.data)
            elif ecode.shape[0] > 1:
                raise Exception(f'ERROR: {ecode.shape[0]} instances of {c} in trial {t_idx}')
        trials[c] = values
        
    # Check that each header code only appears once
    if not (header_ecodes.columns.value_counts().to_numpy() == 1).all():
        raise Exception('ERROR: redundant header codes. Did you press reset states twice?')

    header_dict = {}

    for i in range(header_ecodes.shape[0]):
        h_ecode = header_ecodes.iloc[i]
        header_dict[h_ecode.column] = h_ecode.data


    header = pd.Series(header_dict)

    return header, trials