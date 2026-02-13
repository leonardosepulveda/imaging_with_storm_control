import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Optional, List
from pathlib import Path
from xml.dom import minidom


def get_color_to_channel_dict(microscope='MF3'):
    if microscope=='MF3' or \
       microscope=='MF5':
        d = dict(zip([np.nan, 405,488,560,650,750],[np.nan,4,3,2,1,0]))
    return d

def get_frame_table(bead_z , bead_seq, color_seq , end_seq, z_pos, microscope='MF3'):
    """
    For simplicity, the shutter files have the following features:
        - fixed number of frames per z position (to have a nZ x m frames)
        - first z position is z = 0
        - last z position is z = 0
        - blank frames have np.nan color
    
    """
   
    color_2_channel_dict = get_color_to_channel_dict(microscope=microscope)

    data = []

    # add frames for the imaging of beads at z == bead_z
    for i , color in enumerate(bead_seq):
        data.append([color, color_2_channel_dict[color], bead_z])

    # add frames for the imaging of sample
    for z in z_pos:
        for c in color_seq:
            data.append([c, color_2_channel_dict[c], z])

    # add frames for the imaging of beads at z == bead_z
    for i , color in enumerate(end_seq):
        data.append([color, color_2_channel_dict[color], bead_z])

    frame_df = pd.DataFrame(columns=['color', 'channel', 'z'], data=data)
            
    return frame_df

def print_frame_table(df):
    """
    To easily visualize the structure of the shutter file
    """
    
    # extract the number of frames per z
    counts = df['z'].value_counts()
    frames_per_z = min(counts)
    
    col_width = 6
    col_sep = ' '*6
    
    # Header
    header_width = col_width * frames_per_z
    print(f'{"frames":{header_width}s}{col_sep}'
          f'{"color":{header_width}s}{col_sep}'
          f'{"channel":{header_width}s}{col_sep}'
          f'{"z":{header_width}s}{col_sep}')
    print()  # blank line

    n = len(df)

    for start in range(0, n, frames_per_z):
        end = start + frames_per_z
        group = df.iloc[start:end]

        # Skip incomplete group if desired; remove this if you want to print it
        if len(group) < frames_per_z:
            break

        # Frames (index)
        frames_str = ''.join(f'{int(idx):{col_width}d}' for idx in group.index)

        # Helper for NaNs in int-like columns
        def fmt_int_like(val):
            if pd.isna(val):
                return f'{"nan":>{col_width}}'
            return f'{int(val):{col_width}d}'

        # Colors
        colors_str = ''.join(fmt_int_like(v) for v in group['color'])

        # Channels
        channels_str = ''.join(fmt_int_like(v) for v in group['channel'])

        # z as float with 2 decimals
        z_str = ''.join(f'{float(v):{col_width}.2f}' for v in group['z'])

        print(f'{frames_str}{col_sep}{colors_str}{col_sep}{channels_str}{col_sep}{z_str}')
        
def get_color_sequence_name(df):
    """
    Given a DataFrame with a 'color' column, return a succinct name like:
    'blkf8_405f2_650f4_750f4'
    where:
      - 'blk' corresponds to NaN (blank) entries
      - other entries are color values (as ints)
      - 'fN' is the total count of frames with that color
    """
    col = df['color']

    # Count NaNs separately
    n_blk = col.isna().sum()

    # Count non-NaN colors
    counts = col.value_counts(dropna=True)

    # Build parts of the name
    parts = []

    if n_blk > 0:
        parts.append(f"blkf{int(n_blk)}")

    # Sort colors numerically, if they are numeric
    # (convert index to float, then int for printing)
    for color in sorted(counts.index.astype(float)):
        count = int(counts.loc[color])
        parts.append(f"{int(color)}f{count}")

    # Join with underscores
    return "_".join(parts)

def create_shutter_file(df, filename, oversampling=1, default_power=1.0):
    """
    Convert a DataFrame with columns: ['color', 'channel', 'z']
    index = frame number (on-time)
    to an XML file of events, pretty-printed with CRLF line endings.

    Additionally:
      - Embed the full frame table (as CSV) into the first XML comment so
        that the original z (and full frame info) can be reconstructed
        later from the shutter file alone.
    """
    # Ensure index is numeric and sorted
    df = df.sort_index()

    # Root element
    root = ET.Element("repeat")

#     # --- Add frame table as CSV in a comment at the beginning ---
#     # We include the index as "frame"
#     csv_buf = io.StringIO()
#     df.to_csv(csv_buf, index=True)   # index name will become first column header
#     csv_text = csv_buf.getvalue()

#     frame_table_comment_text = (
#         "FRAME_TABLE_CSV_START\n"
#         + csv_text +
#         "FRAME_TABLE_CSV_END"
#     )

#     comment_el = ET.Comment(frame_table_comment_text)
#     root.append(comment_el)
#     # Two empty lines after the comment, before <oversampling>...
#     # (minidom/toprettyxml will preserve this as blank lines)
#     comment_el.tail = "\n\n"

    # <oversampling>
    overs_el = ET.SubElement(root, "oversampling")
    overs_el.text = str(oversampling)

    # <frames>
    frames_el = ET.SubElement(root, "frames")
    frames_el.text = str(len(df))

    last_z = None  # to track when z changes and add a comment

    for frame, row in df.iterrows():
        channel = row["channel"]
        z = row["z"]
        color = row.get("color", np.nan)  # in case 'color' column might be missing

        # Skip non-event frames (NaN channel)
        if pd.isna(channel):
            continue

        # Add comment when z changes (or for the first event)
        if (last_z is None) or (z != last_z):
            if z == 0 and not pd.isna(color) and int(color) == 405:
                comment_text = f" z = {int(z)} um, 405 beads"
            else:
                if float(z).is_integer():
                    comment_text = f" z = {int(z)} um"
                else:
                    comment_text = f" z = {z} um"
            root.append(ET.Comment(comment_text))
            last_z = z

        # <event>
        event_el = ET.SubElement(root, "event")

        ch_el = ET.SubElement(event_el, "channel")
        ch_el.text = str(int(channel))

        pw_el = ET.SubElement(event_el, "power")
        pw_el.text = f"{default_power:.1f}"

        on_el = ET.SubElement(event_el, "on")
        on_el.text = f"{float(frame):.1f}"

        off_el = ET.SubElement(event_el, "off")
        off_el.text = f"{float(frame + 1):.1f}"

    # ---- Pretty-print + CRLF writing ----
    rough_bytes = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(rough_bytes)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="ISO-8859-1")  # bytes

    # Decode to text (LF newlines for now)
    pretty_text = pretty_xml.decode("ISO-8859-1")

    # Add an empty line before every comment line for legibility
    pretty_text = pretty_text.replace("\n  <!--", "\n\n  <!--")

    # Normalize to CRLF for Windows
    pretty_text = pretty_text.replace("\n", "\r\n")

    with open(filename, "w", encoding="ISO-8859-1", newline="") as f:
        f.write(pretty_text)
        
def format_z_offsets_from_frame_table(df: pd.DataFrame) -> str:
    """
    Build the text for <z_offsets> from the 'z' column of frame_table.

    Logic (matches your example):
      - Determine group size as the length of a contiguous run of identical z
        (assumes all runs have same length).
      - Take one z per run (i.e. 0.0, 1.0, 1.5, 2.0, 2.5, 0.0 in your example).
      - For each such z, repeat it group_size times.
      - Arrange values in rows of 'group_size' values.
      - Add a comma after every value except the very last overall value.
    """
    z = df['z'].astype(float).to_list()

    # Compute run lengths to infer group size
    run_lengths = []
    last = None
    count = 0
    for val in z:
        if last is None or val == last:
            count += 1
        else:
            run_lengths.append(count)
            count = 1
        last = val
    if count:
        run_lengths.append(count)

    group_size = run_lengths[0]
    # (Optional sanity check)
    if not all(rl == group_size for rl in run_lengths):
        raise ValueError(f"Inconsistent run lengths in z column: {run_lengths}")

    # Unique z per run (in order)
    unique_z_per_run = []
    last = object()
    for val in z:
        if val != last:
            unique_z_per_run.append(val)
            last = val

    # Build flat list: each run's z repeated group_size times
    values = []
    for val in unique_z_per_run:
        values.extend([val] * group_size)

    # Now format as text for the XML, with commas as specified
    lines = []
    total = len(values)

    for i, val in enumerate(values):
        is_last = (i == total - 1)
        suffix = '' if is_last else ','  # no comma after very last value
        token = f"{val:.1f}{suffix}"

        row_idx = i // group_size
        if len(lines) <= row_idx:
            lines.append([])
        lines[row_idx].append(token)

    # Indentation is mainly cosmetic; important part is comma placement
    # We’ll match your style roughly:
    #   0.0,  0.0,  0.0,
    # etc., with two spaces between tokens.
    indent = "         "  # 9 spaces, as in your example
    inner_lines = []
    for row in lines:
        inner_lines.append(indent + "  ".join(row))

    # Surround with newlines so ElementTree pretty-prints reasonably.
    text = "\n" + "\n".join(inner_lines) + "\n      "
    return text

