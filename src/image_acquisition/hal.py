import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Optional, List
from pathlib import Path

def create_hal_config(prefix: str,
                      frame_table: pd.DataFrame,
                      default_power: Optional[List[float]] = None,
                      xml_dir: str or Path = ".",
                      output_dir: str or Path = ".") -> Path:
    """
    Read '{prefix}.xml' as plain text from xml_dir, update:
      - <frames>            -> len(frame_table)
      - <default_power>     -> comma-joined default_power (if provided)
      - <shutters>          -> '{color_sequence_name}_shutter.xml'
      - <z_offsets>         -> formatted from frame_table['z']

    Then write '{output_dir}/{prefix}-{color_sequence_name}.xml',
    preserving original comments and overall layout as much as possible,
    and adding exactly one empty line before each comment for legibility.
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)

    in_path = xml_dir / f"{prefix}.xml"
    with open(in_path, "rb") as f:
        raw = f.read()

    # Decode and normalize to '\n' internally
    text = raw.decode("ISO-8859-1").replace("\r\n", "\n")

    # --- 1. frames ---
    def repl_frames(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{len(frame_table)}{close_tag}"

    text = re.sub(
        r"(<frames[^>]*>)(.*?)(</frames>)",
        repl_frames,
        text,
        flags=re.DOTALL,
    )

    # --- 2. default_power (optional) ---
    if default_power is not None:
        dp_str = ",".join(str(v) for v in default_power)

        def repl_default_power(m):
            open_tag, _, close_tag = m.groups()
            return f"{open_tag}{dp_str}{close_tag}"

        text = re.sub(
            r"(<default_power[^>]*>)(.*?)(</default_power>)",
            repl_default_power,
            text,
            flags=re.DOTALL,
        )

    # --- 3. shutters ---
    seq_name = get_color_sequence_name(frame_table)
    shutters_value = f"{seq_name}_shutter.xml"

    def repl_shutters(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{shutters_value}{close_tag}"

    text = re.sub(
        r"(<shutters[^>]*>)(.*?)(</shutters>)",
        repl_shutters,
        text,
        flags=re.DOTALL,
    )

    # --- 4. z_offsets from frame_table['z'] ---
    new_z_block = format_z_offsets_from_frame_table(frame_table)
    # new_z_block should already contain leading/trailing newlines and indentation

    def repl_z_offsets(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{new_z_block}{close_tag}"

    text = re.sub(
        r"(<z_offsets[^>]*>)(.*?)(</z_offsets>)",
        repl_z_offsets,
        text,
        flags=re.DOTALL,
    )

    # --- 5. Ensure exactly one empty line before each comment ---
    # Only add an extra blank line if there isn't already one just before the comment.
    # Preserve indentation before <!-- by capturing spaces/tabs only.
    text = re.sub(
        r"(?<!\n\n)\n([ \t]*)<!--",   # a newline, optional spaces/tabs, then <!--
        r"\n\n\1<!--",
        text,
    )

    # --- 6. Write out with CRLF line endings ---
    out_path = output_dir / f"{prefix}-{seq_name}.xml"
    with open(out_path, "w", encoding="ISO-8859-1", newline="") as f:
        f.write(text.replace("\n", "\r\n"))

    return out_path

def _strip_whitespace_etree(elem):
    """
    Remove whitespace-only .text and .tail from an ElementTree tree.
    This prevents minidom.toprettyxml from inserting extra blank lines.
    """
    # Clean children first
    for child in list(elem):
        _strip_whitespace_etree(child)
        if child.tail is not None and not child.tail.strip():
            child.tail = None

    # Clean this element's text
    if elem.text is not None and not elem.text.strip():
        elem.text = None
        
    
def print_xml_raw(path, encoding="ISO-8859-1"):
    """
    Print the XML file exactly as it is on disk.
    - Comments are preserved.
    - Existing indentation and spacing are preserved.
    - CRLF line endings are preserved (or optionally normalized).
    """
    with open(path, "rb") as f:
        data = f.read()

    # Option A: preserve original CRLF exactly
    text = data.decode(encoding)
    print(text, end="")  # no extra newline at end

    # If you prefer normalized LF for your console instead, use:
    # text = data.decode(encoding).replace("\r\n", "\n")
    # print(text, end="")
    

# def read_shutter_file_to_frame_table(filename, microscope='MF3'):
#     """
#     Recreate a frame table from a shutter XML file created by `create_shutter_file`.

#     Logic:
#       - The first XML comment contains the full frame table as CSV, between
#         the markers 'FRAME_TABLE_CSV_START' and 'FRAME_TABLE_CSV_END'.
#       - We parse that CSV to recover z for every frame (and can also see
#         color/channel, if desired).
#       - We then parse <event> elements to rebuild the actual color/channel
#         activity per frame (overwriting any color/channel from the CSV
#         with what the events say).
#       - Result:
#           * z comes from the first comment's CSV.
#           * color & channel come from the events (non-event frames get NaN).

#     Returns
#     -------
#     frame_df : pandas.DataFrame
#         Columns: ['color', 'channel', 'z'], indexed by frame number.
#     """
#     # --- 1. Parse XML with comments preserved ---
#     parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
#     tree = ET.parse(filename, parser=parser)
#     root = tree.getroot()   # <repeat>

#     # --- 2. Read total number of frames (for sanity) ---
#     frames_el = root.find("frames")
#     if frames_el is None or frames_el.text is None:
#         raise ValueError("Could not find <frames> element in shutter file.")
#     n_frames = int(float(frames_el.text))

#     # --- 3. Extract frame table CSV from the first comment ---
#     csv_text = None
#     for child in root:
#         if child.tag is Comment and child.text:
#             txt = child.text
#             if "FRAME_TABLE_CSV_START" in txt and "FRAME_TABLE_CSV_END" in txt:
#                 # Extract between the markers
#                 start = txt.index("FRAME_TABLE_CSV_START") + len("FRAME_TABLE_CSV_START")
#                 end = txt.index("FRAME_TABLE_CSV_END")
#                 csv_body = txt[start:end].strip("\n")
#                 csv_text = csv_body
#                 break

#     if csv_text is None:
#         raise ValueError("Could not find FRAME_TABLE_CSV comment in shutter file.")

#     # Parse CSV into DataFrame
#     csv_buf = io.StringIO(csv_text)
#     # The first column should be "frame" (index); we force that to be index_col=0
#     df_csv = pd.read_csv(csv_buf, index_col=0)
#     df_csv.index.name = "frame"

#     # Sanity check length
#     if len(df_csv) != n_frames:
#         # Not fatal, but warn user
#         print(f"Warning: frames in CSV ({len(df_csv)}) != <frames> ({n_frames})")

#     # We will take z from the CSV, but color/channel from events
#     z_series = df_csv["z"].astype(float)

#     # --- 4. Build inverse color<->channel mapping (in case needed) ---
#     color_to_channel = get_color_to_channel_dict(microscope=microscope)
#     channel_to_color = {}
#     for col, ch in color_to_channel.items():
#         if pd.isna(col) or pd.isna(ch):
#             continue
#         channel_to_color[int(ch)] = float(col)

#     # --- 5. Walk children again, reading <event> elements ---
#     events = {}

#     for child in root:
#         if child.tag != "event":
#             continue

#         on_el = child.find("on")
#         ch_el = child.find("channel")
#         if on_el is None or ch_el is None:
#             continue

#         frame = int(float(on_el.text))
#         channel = int(ch_el.text)

#         # Color from channel
#         color = channel_to_color.get(channel, np.nan)

#         events[frame] = (color, float(channel))

#     # --- 6. Build full frame table using:
#     #       - z from comment-CSV
#     #       - color/channel from events
#     #       - NaNs where there is no event
#     data = []
#     max_frame = max(n_frames, len(z_series))
#     for f in range(max_frame):
#         z = z_series.loc[f] if f in z_series.index else np.nan

#         if f in events:
#             color, channel = events[f]
#         else:
#             color = np.nan
#             channel = np.nan

#         data.append([color, channel, z])

#     frame_df = pd.DataFrame(data, columns=["color", "channel", "z"])
#     frame_df.index.name = "frame"
#     return frame_df

