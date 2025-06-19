import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO # To read uploaded CSV data
import altair as alt # Import Altair for interactive charts

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Cell Congestion Analysis Dashboard",
    layout="wide", # Use wide layout for more space for charts and table
    initial_sidebar_state="expanded"
)

st.title("📡 Cell Congestion Analysis Dashboard")
st.markdown("Upload your CSV file(s) to analyze cell and sector congestion. This dashboard provides a detailed breakdown of cell performance and overall network health.")

# --- Helper Functions (defined at the top for proper scope) ---

def get_top_n_avg(series, n=4):
    """
    Function to get the average of the top N highest values in a Series.
    Handles cases where there are fewer than N valid values gracefully by returning NaN.
    If all values are zero, it will correctly return 0.
    """
    series = series.dropna() # Drop NaNs before finding largest
    if len(series) == 0:
        return np.nan
    top_values = series.nlargest(n) # Get the N largest values
    if top_values.empty:
        return np.nan
    return top_values.mean() # Calculate their average

def get_weekend_value(series):
    """
    Computes the weekend value based on specific rules for two weekend days:
    - Only considers non-zero values for average.
    - If only one non-zero value, use that value.
    - If both are non-zero, average them.
    - Returns NaN if no non-zero weekend values.
    """
    series_non_zero_weekend = series.dropna()[series.dropna() != 0] # Filter for non-NaN and non-zero values
    num_non_zero_weekend = len(series_non_zero_weekend)

    if num_non_zero_weekend == 0:
        return np.nan
    elif num_non_zero_weekend == 1:
        return series_non_zero_weekend.iloc[0]
    else: # num_non_zero_weekend >= 2 (implies 2 non-zero values for weekends)
        return series_non_zero_weekend.mean()

def compute_all_days_value(series):
    """
    Computes the 'all days' value based on non-zero values and count.
    - If less than 4 non-zero values, average them.
    - If 4 or more non-zero values, average the top 4 highest non-zero values.
    - Returns NaN if no non-zero values.
    """
    series_non_zero = series.dropna()[series.dropna() != 0] # Filter for non-NaN and non-zero values
    count_non_zero = len(series_non_zero)

    if count_non_zero == 0:
        return np.nan
    elif count_non_zero < 4:
        return series_non_zero.mean()
    else: # count_non_zero >= 4
        return series_non_zero.nlargest(4).mean()

def parse_cell_name_refined(cell_name):
    sitename = np.nan
    band = np.nan
    sector_id = np.nan
    sector_name = np.nan

    if not isinstance(cell_name, str) or not cell_name:
        return sitename, band, sector_id, sector_name

    temp_cell_name = cell_name.strip() # Start by stripping whitespace

    # Define suffixes to remove using a single regex pattern for efficiency.
    # Added '_RSS', ' 4TRFS' (with space), ' 4THES' (with space), and ensuring flexibility for digits/letters.
    # The pattern now also includes a general match for _ followed by numbers/letters at the end.
    suffixes_to_remove_pattern = r'(?:_4RFS|_4TRFS|_INTERIM|_324M|_CSP|__4RFS|_RSS| 4TRFS| 4THES|_R\d+|_L\d+)$'
    temp_cell_name = re.sub(suffixes_to_remove_pattern, '', temp_cell_name)

    # Find the last hyphen. This is crucial for separating sitename/band from sector ID part.
    last_hyphen_pos = temp_cell_name.rfind('-')

    if last_hyphen_pos == -1:
        # If no hyphen is found in the cleaned name, this format might not be supported yet.
        return np.nan, np.nan, np.nan, np.nan

    # --- Extract SECTORID ---
    # The part of the string after the last hyphen (e.g., "134", "L91", "2")
    part_after_hyphen = temp_cell_name[last_hyphen_pos + 1:]
    
    # Find all digits in the part after the hyphen.
    # The sector_id is "the very last digit on the number after the dash (-)".
    # This implies we look for any digit in this segment.
    all_digits_in_part_after_hyphen = re.findall(r'\d', part_after_hyphen)
    
    if all_digits_in_part_after_hyphen:
        sector_id = all_digits_in_part_after_hyphen[-1] # Take the very last digit found
    else:
        # If no digit is found after the hyphen, sector ID cannot be determined.
        return np.nan, np.nan, np.nan, np.nan # Early exit as sector_id is essential

    # Now, the part before the last hyphen is `sitename_band_part`
    sitename_band_part = temp_cell_name[:last_hyphen_pos]

    # --- Extract BAND and SITENAME ---
    band_found_in_special_case = False

    # Special Case: Handle the '_D' pattern (e.g., "GTCANDUMMNDAUECEBV_D")
    # If `sitename_band_part` ends with '_D', the band is the character before '_D'.
    if sitename_band_part.endswith('_D'):
        # Check if there is a capital letter immediately before '_D'
        if len(sitename_band_part) >= 3 and sitename_band_part[-3].isupper():
            band = sitename_band_part[-3] # The character before '_D'
            # SITENAME is everything before the band, then append '_D' back.
            sitename_candidate = sitename_band_part[:-3]
            sitename = sitename_candidate + '_D'
            band_found_in_special_case = True
    
    if not band_found_in_special_case:
        # General case: last capital letter in `sitename_band_part` is the band.
        # This handles formats like "SITENAMEBAND" or "SITENAMEXXXBAND" (where XXX are digits/chars)
        all_caps_in_sitename_band_part = re.findall(r'[A-Z]', sitename_band_part)
        if all_caps_in_sitename_band_part:
            band = all_caps_in_sitename_band_part[-1] # The last capital letter is the band
            band_pos = sitename_band_part.rfind(band)
            sitename_candidate = sitename_band_part[:band_pos] # SITENAME is everything before that last capital letter
            sitename = sitename_candidate
        else:
            # If no capital letter found before the hyphen, band cannot be reliably determined.
            band = np.nan
            sitename = sitename_band_part # Assume the whole pre-hyphen part is sitename, but band is unknown.


    # Final cleaning of SITENAME: remove any leading/trailing non-alphanumeric characters.
    # Allowing underscore in sitename as per previous discussions (e.g., GTCANDUMMNDAUECEB_D).
    if pd.notna(sitename) and isinstance(sitename, str):
        # Remove leading non-alphanumeric chars (e.g., '.' in '.TCFOAVAREVALOILOILOILOW')
        sitename = re.sub(r'^[^a-zA-Z0-9_]+', '', sitename)
        # Remove trailing non-alphanumeric chars (e.g., if there were any accidental trailing symbols)
        sitename = re.sub(r'[^a-zA-Z0-9_]+$', '', sitename)
        if not sitename.strip(): # If it becomes empty after stripping, make it NaN
            sitename = np.nan


    # --- Construct SECTORNAME ---
    # SECTORNAME is concatenation of SITENAME and SECTORID, only if both are valid.
    if pd.notna(sitename) and pd.notna(sector_id):
        sector_name = f"{sitename}{sector_id}"
    else:
        sector_name = np.nan

    return sitename, band, sector_id, sector_name


def calculate_traffic_score(row):
    traffic_user = row['Final Traffic']
    band = row['BAND']

    if pd.isna(traffic_user):
        return np.nan

    if band in ['F', 'W', 'G', 'Q', 'Y']: # 10MHZ
        if traffic_user > 129: return 4
        if traffic_user > 99: return 3
        if traffic_user > 59: return 2
        if traffic_user == 0: return 0
        return 1
    elif band == 'L': # 15MHZ
        if traffic_user > 169: return 4
        if traffic_user > 119: return 3
        if traffic_user > 67: return 2
        if traffic_user == 0: return 0
        return 1
    elif band == 'K': # 15MHZ
        if traffic_user > 134: return 4
        if traffic_user > 89: return 3
        if traffic_user > 52: return 2
        if traffic_user == 0: return 0
        return 1
    elif band == 'H': # 20MHZ
        if traffic_user > 179: return 4
        if traffic_user > 119: return 3
        if traffic_user > 69: return 2
        if traffic_user == 0: return 0
        return 1
    elif band == 'V': # 20MHZ
        if traffic_user > 719: return 4
        if traffic_user > 479: return 3
        if traffic_user > 279: return 2
        if traffic_user == 0: return 0
        return 1

    return np.nan

def calculate_prb_score(prb_dl):
    if pd.isna(prb_dl):
        return np.nan

    if prb_dl > 79: return 4
    if prb_dl > 59: return 3
    if prb_dl > 24: return 2
    if prb_dl == 0: return 0
    return 1

def get_color_code(row, color_matrix):
    concat_score = row['Concatenated Score']
    bandwidth = row['BANDWIDTH']

    if pd.isna(concat_score) or pd.isna(bandwidth):
        return np.nan

    lookup_key = (concat_score, bandwidth)

    return color_matrix.get(lookup_key, 'UNKNOWN_COLOR')


def get_final_assessment(row):
    """
    Determines the FINAL ASSESSMENT for a cell based on its Cell Level Congestion
    and the sector-wide counts of Red and Good cells.
    """
    cell_level_congestion = row['Cell Level Congestion']
    good_cells_count_in_sector = row['Good_Cells_Count_Sector'] # Using new column name from merge
    red_cells_count_in_sector = row['Red_Cells_Count_Sector']   # Using new column name from merge

    if pd.isna(cell_level_congestion) or pd.isna(good_cells_count_in_sector) or pd.isna(red_cells_count_in_sector):
        return "-" # Default for missing data

    # New logic based on your clarification:
    # If a certain cell is RED, then it checks if Good_cells_count_in_sector >= Red_cells_count_in_sector.
    # If true, returns "FOR TRAFFIC BALANCING".
    # Otherwise (if RED cell AND Good < Red), it returns "NEED EXPANSION".
    if cell_level_congestion == 'RED':
        if good_cells_count_in_sector >= red_cells_count_in_sector:
            return "FOR TRAFFIC BALANCING"
        else: # cell_level_congestion is RED AND red_cells_count_in_sector > good_cells_count_in_sector
            return "NEED EXPANSION"
    # If the cell itself is NOT RED (e.g., Green, Amber, Yellow, Blue, Black, Unknown)
    # The condition for these is ONLY 'NEED EXPANSION' if red_cells_count_in_sector > good_cells_count_in_sector,
    # otherwise, it's 'NOT CONGESTED'.
    else: 
        if red_cells_count_in_sector > good_cells_count_in_sector:
            return "NEED EXPANSION"
        # If not "NEED EXPANSION", it's "NOT CONGESTED" for non-red cells in good/balanced sectors
        else: 
            return "NOT CONGESTED"


def get_cell_to_sector_color_coding(row):
    black_count = row['Black_Cells_Count']
    red_count = row['Red_Cells_Count_for_CTSCC']
    amber_count = row['Amber_Cells_Count']
    total_count = row['Total_Cells_in_Sector_Band']

    if pd.isna(total_count) or total_count == 0:
        return np.nan # No cells in this group to assess

    # Condition 1: If BLACK CELLS/TOTAL = 100%
    if black_count == total_count:
        return "BLACK"

    # Omit black cells for subsequent conditions
    effective_total = total_count - black_count
    if effective_total <= 0: # If all non-black cells were effectively removed (e.g., all black or only NaN colors)
        return np.nan # Or a specific indicator if you prefer, like "N/A_NON_BLACK"

    # Condition 2: IF RED CELLS/TOTAL > 50% (using effective_total)
    if (red_count / effective_total) > 0.5:
        return "RED"

    # Condition 3: IF (RED+AMBER)/TOTAL >= 50% (using effective_total)
    if ((red_count + amber_count) / effective_total) >= 0.5:
        return "AMBER"
        
    # Fallback: If none of the above conditions are met, it's GREEN
    return "GREEN"


# --- Global Constants ---
bandwidth_map = {
    'F': '10MHZ', 'W': '10MHZ', 'G': '10MHZ', 'Q': '10MHZ', 'Y': '10MHZ',
    'L': '15MHZ', 'K': '15MHZ',
    'H': '20MHZ', 'V': '20MHZ'
}

color_matrix = {
    ('44', '20MHZ'): 'RED', ('44', '15MHZ'): 'RED', ('44', '10MHZ'): 'RED', ('44', '5MHZ'): 'RED',
    ('43', '20MHZ'): 'AMBER', ('43', '15MHZ'): 'RED', ('43', '10MHZ'): 'RED', ('43', '5MHZ'): 'RED',
    ('42', '20MHZ'): 'YELLOW', ('42', '15MHZ'): 'AMBER', ('42', '10MHZ'): 'AMBER', ('42', '5MHZ'): 'RED',
    ('41', '20MHZ'): 'YELLOW', ('41', '15MHZ'): 'AMBER', ('41', '10MHZ'): 'AMBER', ('41', '5MHZ'): 'RED',
    ('34', '20MHZ'): 'YELLOW', ('34', '15MHZ'): 'YELLOW', ('34', '10MHZ'): 'YELLOW', ('34', '5MHZ'): 'AMBER',
    ('33', '20MHZ'): 'GREEN', ('33', '15MHZ'): 'GREEN', ('33', '10MHZ'): 'YELLOW', ('33', '5MHZ'): 'AMBER',
    ('32', '20MHZ'): 'GREEN', ('32', '15MHZ'): 'GREEN', ('32', '10MHZ'): 'YELLOW', ('32', '5MHZ'): 'AMBER',
    ('31', '20MHZ'): 'GREEN', ('31', '15MHZ'): 'GREEN', ('31', '10MHZ'): 'YELLOW', ('31', '5MHZ'): 'AMBER',
    ('24', '20MHZ'): 'GREEN', ('24', '15MHZ'): 'GREEN', ('24', '10MHZ'): 'GREEN', ('24', '5MHZ'): 'YELLOW',
    ('23', '20MHZ'): 'GREEN', ('23', '15MHZ'): 'GREEN', ('23', '10MHZ'): 'GREEN', ('23', '5MHZ'): 'YELLOW',
    ('22', '20MHZ'): 'GREEN', ('22', '15MHZ'): 'GREEN', ('22', '10MHZ'): 'GREEN', ('22', '5MHZ'): 'YELLOW',
    ('21', '20MHZ'): 'GREEN', ('21', '15MHZ'): 'GREEN', ('21', '10MHZ'): 'GREEN', ('21', '5MHZ'): 'YELLOW',
    ('14', '20MHZ'): 'BLUE', ('14', '15MHZ'): 'BLUE', ('14', '10MHZ'): 'GREEN', ('14', '5MHZ'): 'GREEN',
    ('13', '20MHZ'): 'BLUE', ('13', '15MHZ'): 'BLUE', ('13', '10MHZ'): 'GREEN', ('13', '5MHZ'): 'GREEN',
    ('12', '20MHZ'): 'BLUE', ('12', '15MHZ'): 'BLUE', ('12', '10MHZ'): 'GREEN', ('12', '5MHZ'): 'GREEN',
    ('11', '20MHZ'): 'BLUE', ('11', '15MHZ'): 'BLUE', ('11', '10MHZ'): 'GREEN', ('11', '5MHZ'): 'GREEN',
    ('00', '20MHZ'): 'BLACK', ('00', '15MHZ'): 'BLACK', ('00', '10MHZ'): 'BLACK', ('00', '5MHZ'): 'BLACK',
    ('10', '20MHZ'): 'GREEN', ('10', '15MHZ'): 'GREEN', ('10', '10MHZ'): 'GREEN', ('10', '5MHZ'): 'GREEN',
    ('01', '20MHZ'): 'GREEN', ('01', '15MHZ'): 'GREEN', ('01', '10MHZ'): 'GREEN', ('01', '5MHZ'): 'GREEN',
}


@st.cache_data(show_spinner="Processing data and running analysis...")
def process_congestion_data(raw_df):
    """
    Performs all data cleaning, KPI computation, parsing, scoring, and merging
    to produce the final analysis DataFrame. This function is cached.
    """
    df = raw_df.copy() # Work on a copy to avoid modifying the original DataFrame in cache

    # --- Ensure all required columns exist with placeholders at the very beginning ---
    required_cols = ['Date', 'CATEGORY', 'Province', 'Town', 'Vendor']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 'N/A' # Default to 'N/A' if column is entirely missing
            st.warning(f"Column '{col}' not found in the uploaded file. Using 'N/A' as placeholder.")

    # --- Automate CATEGORY based on Date column (now 'Date' and 'CATEGORY' are guaranteed to exist) ---
    # Check if the 'Date' column itself is not entirely 'N/A'
    if not df['Date'].astype(str).str.strip().str.lower().eq('n/a').all(): 
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
        original_rows_before_date_dropna = len(df)
        df.dropna(subset=['Date'], inplace=True) # Drop rows where Date could not be parsed
        if len(df) < original_rows_before_date_dropna:
            st.warning(f"Dropped {original_rows_before_date_dropna - len(df)} rows due to unparseable dates.")
        
        df['CATEGORY'] = df['Date'].apply(lambda x: 'weekend' if pd.notna(x) and x.weekday() >= 5 else 'weekday')
        st.success("Date column processed and CATEGORY automated!")
    else: # 'Date' column was missing or all 'N/A', try to use existing CATEGORY or keep 'N/A'
        # Check if existing CATEGORY column has non-'N/A' and non-null values
        if not df['CATEGORY'].astype(str).str.strip().str.lower().eq('n/a').all() and not df['CATEGORY'].isnull().all():
            df['CATEGORY'] = df['CATEGORY'].astype(str).str.lower()
            df.loc[df['CATEGORY'] == 'nan', 'CATEGORY'] = np.nan # Ensure 'nan' string is actual NaN
            st.success("Using existing 'CATEGORY' column.")
        else:
            st.info("Neither valid 'Date' nor existing 'CATEGORY' column found. 'CATEGORY' remains 'N/A'.")


    # --- Capture ALL unique Cell Names and their associated metadata from the ORIGINAL DataFrame first ---
    # All columns used here are guaranteed to exist now due to initial checks
    all_original_cell_names = df[['Cell Name', 'Province', 'Town', 'Vendor']].drop_duplicates().reset_index(drop=True)


    # --- Data Cleaning (continued) ---
    df['PRB_DL_Usage'] = pd.to_numeric(df['PRB_DL_Usage'], errors='coerce')
    df['L.Traffic.User.Avg'] = pd.to_numeric(df['L.Traffic.User.Avg'], errors='coerce')
    df.dropna(subset=['PRB_DL_Usage', 'L.Traffic.User.Avg'], how='all', inplace=True)

    # --- PRB_DL_Usage Computation ---
    all_days_prb_computed = df.groupby('Cell Name')['PRB_DL_Usage'].apply(compute_all_days_value).rename('All_Days_Computed_PRB')
    weekend_only_prb_computed = df[df['CATEGORY'] == 'weekend'].groupby('Cell Name')['PRB_DL_Usage'].apply(get_weekend_value).rename('Weekend_Computed_PRB')
    
    prb_comparison_df = pd.merge(all_days_prb_computed, weekend_only_prb_computed, on='Cell Name', how='outer')

    final_prb_values = {}
    for cell_name in prb_comparison_df.index:
        all_days_val = prb_comparison_df.loc[cell_name, 'All_Days_Computed_PRB']
        weekend_val = prb_comparison_df.loc[cell_name, 'Weekend_Computed_PRB']

        if pd.isna(all_days_val) and pd.isna(weekend_val):
            final_prb_values[cell_name] = np.nan
        elif pd.isna(all_days_val):
            final_prb_values[cell_name] = weekend_val
        elif pd.isna(weekend_val):
            final_prb_values[cell_name] = all_days_val
        else:
            # Explicitly compare non-NaN values
            final_prb_values[cell_name] = all_days_val if all_days_val >= weekend_val else weekend_val
    final_prb_series = pd.Series(final_prb_values, name='Final PRB')

    # --- L.Traffic.User.Avg Computation ---
    all_days_traffic_computed = df.groupby('Cell Name')['L.Traffic.User.Avg'].apply(compute_all_days_value).rename('All_Days_Computed_Traffic')
    weekend_only_traffic_computed = df[df['CATEGORY'] == 'weekend'].groupby('Cell Name')['L.Traffic.User.Avg'].apply(get_weekend_value).rename('Weekend_Computed_Traffic')
    
    traffic_comparison_df = pd.merge(all_days_traffic_computed, weekend_only_traffic_computed, on='Cell Name', how='outer')

    final_traffic_values = {}
    for cell_name in traffic_comparison_df.index:
        all_days_val = traffic_comparison_df.loc[cell_name, 'All_Days_Computed_Traffic']
        weekend_val = traffic_comparison_df.loc[cell_name, 'Weekend_Computed_Traffic']

        if pd.isna(all_days_val) and pd.isna(weekend_val):
            final_traffic_values[cell_name] = np.nan
        elif pd.isna(all_days_val):
            final_traffic_values[cell_name] = weekend_val
        elif pd.isna(weekend_val):
            final_traffic_values[cell_name] = all_days_val
        else:
            # Explicitly compare non-NaN values
            final_traffic_values[cell_name] = all_days_val if all_days_val >= weekend_val else weekend_val
    final_traffic_series = pd.Series(final_traffic_values, name='Final Traffic')

    # --- Combine Final PRB and Traffic Results with all original Cell Names, Province, and Town ---
    computed_metrics = pd.DataFrame({
        'Final PRB': final_prb_series,
        'Final Traffic': final_traffic_series
    }).reset_index().rename(columns={'index': 'Cell Name'})

    # Merge `Province`, `Town`, and `Vendor` from `all_original_cell_names`
    final_results_df = pd.merge(all_original_cell_names, computed_metrics, on='Cell Name', how='left')


    # --- Derive SITENAME, BAND, SECTORID, SECTORNAME, BANDWIDTH ---
    final_results_df[['SITENAME', 'BAND', 'SECTORID', 'SECTORNAME']] = \
        final_results_df['Cell Name'].apply(lambda x: pd.Series(parse_cell_name_refined(x)))
    final_results_df['BANDWIDTH'] = final_results_df['BAND'].map(bandwidth_map)
    final_results_df['BANDWIDTH'].fillna('Unknown', inplace=True)

    # --- Calculate L.Traffic.User.Avg Score ---
    final_results_df['L.Traffic.User.Avg Score'] = final_results_df.apply(calculate_traffic_score, axis=1)

    # --- Calculate PRB_DL_Usage Score ---
    final_results_df['PRB_DL_Usage Score'] = final_results_df['Final PRB'].apply(calculate_prb_score)

    # --- Color Coding based on Score Matrix ---
    final_results_df['Concatenated Score'] = final_results_df.apply(
        lambda row: f"{int(row['PRB_DL_Usage Score'])}{int(row['L.Traffic.User.Avg Score'])}"
        if pd.notna(row['PRB_DL_Usage Score']) and pd.notna(row['L.Traffic.User.Avg Score'])
        else np.nan, axis=1
    )
    final_results_df['Color Code'] = final_results_df.apply(lambda row: get_color_code(row, color_matrix), axis=1)
    final_results_df['Color Code'] = final_results_df['Color Code'].apply(lambda x: x.upper() if pd.notna(x) else x)

    # --- RENAME 'Color Code' to 'Cell Level Congestion' ---
    final_results_df.rename(columns={
        'Color Code': 'Cell Level Congestion',
    }, inplace=True)

    # --- Prepare for FINAL ASSESSMENT (Sector-level counts) ---
    # These calculations now use the already renamed 'Cell Level Congestion'
    final_results_df['Is_Good_Cell'] = final_results_df['Cell Level Congestion'].isin(['BLUE', 'GREEN'])
    final_results_df['Is_Red_Cell'] = final_results_df['Cell Level Congestion'] == 'RED' # Keep this for Red_Cells_Count_Sector

    # Aggregate Good and Red cell counts per sector based on unique cells
    sector_level_agg_data = final_results_df.drop_duplicates(subset=['Cell Name', 'SECTORNAME']).groupby('SECTORNAME').agg(
        Good_Cells_Count_Sector=('Is_Good_Cell', 'sum'),
        Red_Cells_Count_Sector=('Is_Red_Cell', 'sum')
    ).reset_index()

    # Merge these sector-wide counts back to the main DataFrame
    final_results_df = pd.merge(
        final_results_df,
        sector_level_agg_data[['SECTORNAME', 'Good_Cells_Count_Sector', 'Red_Cells_Count_Sector']],
        on='SECTORNAME',
        how='left'
    )
    
    # --- FINAL ASSESSMENT Column (now 'Cell Level Congestion' exists and counts are merged) ---
    final_results_df['FINAL ASSESSMENT'] = final_results_df.apply(get_final_assessment, axis=1)

    # --- CELL TO SECTOR COLOR CODING Column ---
    # Ensure these are based on the correct 'Cell Level Congestion' column
    final_results_df['Is_Black_Cell'] = final_results_df['Cell Level Congestion'] == 'BLACK'
    final_results_df['Is_Amber_Cell'] = final_results_df['Cell Level Congestion'] == 'AMBER'
    
    # Re-aggregate counts for sector_band_color_counts based on Cell Level Congestion (using unique cells per sector/band)
    sector_band_color_counts = final_results_df.drop_duplicates(subset=['Cell Name', 'SECTORNAME', 'BAND']).groupby(['SECTORNAME', 'BAND']).agg(
        Black_Cells_Count=('Is_Black_Cell', 'sum'), # Correctly referencing boolean column
        Red_Cells_Count_for_CTSCC=('Is_Red_Cell', 'sum'), # Correctly referencing boolean column
        Amber_Cells_Count=('Is_Amber_Cell', 'sum'), # Correctly referencing boolean column
        Total_Cells_in_Sector_Band=('Cell Name', 'count')
    ).reset_index()

    # Apply get_cell_to_sector_color_coding and then merge
    sector_band_color_counts['Sector Level Congestion Temp'] = sector_band_color_counts.apply(get_cell_to_sector_color_coding, axis=1) 
    
    # Rename 'CELL TO SECTOR COLOR CODING' to 'Sector Level Congestion' AFTER its calculation
    final_results_df = pd.merge(
        final_results_df,
        sector_band_color_counts[['SECTORNAME', 'BAND', 'Sector Level Congestion Temp']],
        on=['SECTORNAME', 'BAND'],
        how='left'
    ).rename(columns={'Sector Level Congestion Temp': 'Sector Level Congestion'})

    # Clean up temporary columns
    final_results_df.drop(columns=[
        'Is_Good_Cell', 'Is_Red_Cell', 'Is_Black_Cell', 'Is_Amber_Cell', # These are temporary boolean flags
    ], inplace=True) 

    # Reorder columns for better readability
    ordered_columns = [
        'Cell Name', 'Province', 'Town', 'Vendor', 'SITENAME', 'BAND', 'BANDWIDTH', 'SECTORID', 'SECTORNAME',
        'Final PRB', 'PRB_DL_Usage Score', 'Final Traffic', 'L.Traffic.User.Avg Score',
        'Concatenated Score', 'Cell Level Congestion',
        'FINAL ASSESSMENT',
        'Sector Level Congestion'
    ]
    return final_results_df[ordered_columns]


# --- Data Loading Section ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file(s)", type=["csv"], accept_multiple_files=False)

raw_df_single_file = None # Initialize raw DataFrame for single file

# Add a button to clear cache and rerun the app
if st.sidebar.button("Clear Cache & Re-run Analysis"):
    st.cache_data.clear()
    st.rerun() 
    
if uploaded_file is not None:
    try:
        # Read the uploaded file into a DataFrame
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        raw_df_single_file = pd.read_csv(stringio)
        st.success(f"File '{uploaded_file.name}' loaded successfully!")
        st.write(f"Initial total rows in raw file: {len(raw_df_single_file)}")
        st.write(f"Initial unique 'Cell Name' count in raw file: {raw_df_single_file['Cell Name'].nunique()}")
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df_single_file.head(), use_container_width=True)
        st.markdown("---")
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file.name}': {e}. Please ensure it's a valid CSV.")
    
# --- Main Logic execution based on the single raw_df ---
if raw_df_single_file is not None:
    try:
        # Process the data using the cached function
        final_results_df = process_congestion_data(raw_df_single_file)
        st.header("2. Data Processing & Analysis")
        st.success("Data processing and analysis complete!")
        # Debugging output: Display intermediate calculation results
        st.subheader("Debugging: Intermediate KPI Computations (Sample)")
        
        # Filter for the specific Cell Name from the example
        # The Cell Name from the image is 'TCPHTMACARTHURILOILOILOF-3_4TRFS'
        example_cell_name = "TCPHTMACARTHURILOILOILOF-3_4TRFS" # Hardcoded for the provided example
        
        # Check if the example cell name exists in the processed data
        if example_cell_name in raw_df_single_file['Cell Name'].unique():
            # Get all relevant data for this cell
            cell_data_for_debug = raw_df_single_file[raw_df_single_file['Cell Name'] == example_cell_name].copy()
            
            # Ensure 'CATEGORY' is correctly populated for debug data
            if 'Date' in cell_data_for_debug.columns:
                cell_data_for_debug['Date'] = pd.to_datetime(cell_data_for_debug['Date'], errors='coerce', infer_datetime_format=True)
                cell_data_for_debug['CATEGORY'] = cell_data_for_debug['Date'].apply(lambda x: 'weekend' if pd.notna(x) and x.weekday() >= 5 else 'weekday')
            elif 'CATEGORY' not in cell_data_for_debug.columns:
                cell_data_for_debug['CATEGORY'] = 'N/A' # Fallback
                
            # Ensure only numeric values are used for debug view
            cell_data_for_debug['PRB_DL_Usage_numeric'] = pd.to_numeric(cell_data_for_debug['PRB_DL_Usage'], errors='coerce')
            cell_data_for_debug['L.Traffic.User.Avg_numeric'] = pd.to_numeric(cell_data_for_debug['L.Traffic.User.Avg'], errors='coerce')

            st.write(f"**Data for Cell: {example_cell_name}**")
            st.dataframe(cell_data_for_debug[['Date', 'CATEGORY', 'L.Traffic.User.Avg_numeric', 'PRB_DL_Usage_numeric']], use_container_width=True)

            # Calculate all-days computed value
            all_days_prb_debug = compute_all_days_value(cell_data_for_debug['PRB_DL_Usage_numeric'])
            all_days_traffic_debug = compute_all_days_value(cell_data_for_debug['L.Traffic.User.Avg_numeric'])
            st.write(f"  - All Days Computed PRB (Top N avg): {all_days_prb_debug}")
            st.write(f"  - All Days Computed Traffic (Top N avg): {all_days_traffic_debug}")

            # Calculate weekend computed value
            weekend_prb_debug = get_weekend_value(cell_data_for_debug[cell_data_for_debug['CATEGORY'] == 'weekend']['PRB_DL_Usage_numeric'])
            weekend_traffic_debug = get_weekend_value(cell_data_for_debug[cell_data_for_debug['CATEGORY'] == 'weekend']['L.Traffic.User.Avg_numeric'])
            st.write(f"  - Weekend Computed PRB: {weekend_prb_debug}")
            st.write(f"  - Weekend Computed Traffic: {weekend_traffic_debug}")

            # Final comparison (using the robust logic from process_congestion_data)
            final_prb_calc = all_days_prb_debug
            if pd.notna(weekend_prb_debug) and (pd.isna(final_prb_calc) or weekend_prb_debug > final_prb_calc):
                final_prb_calc = weekend_prb_debug

            final_traffic_calc = all_days_traffic_debug
            if pd.notna(weekend_traffic_debug) and (pd.isna(final_traffic_calc) or weekend_traffic_debug > final_traffic_calc):
                final_traffic_calc = weekend_traffic_debug

            st.write(f"  - **Final PRB (max): {final_prb_calc}**")
            st.write(f"  - **Final Traffic (max): {final_traffic_calc}**")
        else:
            st.info(f"Sample cell '{example_cell_name}' not found in the uploaded data for debugging.")
        st.markdown("---")

        st.write(f"Total rows in final processed data: {len(final_results_df)}")
        st.write(f"Total unique 'Cell Name' in final processed data: {final_results_df['Cell Name'].nunique()}")


        # --- Display Final Combined Results ---
        st.markdown("---")
        st.header("3. Congestion Analysis Results")

        st.subheader("Final Combined Metrics per Cell")

        # Initialize filtered_df here, unconditionally before any filtering logic
        filtered_df = final_results_df.copy()

        # Filter and Search functionality
        st.sidebar.header("4. Filter & Search Results")
        
        # Text input for searching Cell Name or SITENAME
        search_term = st.sidebar.text_input("Search by Cell Name or SITENAME", "").strip().upper()
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Cell Name'].str.upper().str.contains(search_term, na=False) |
                filtered_df['SITENAME'].str.upper().str.contains(search_term, na=False)
            ]

        # Multi-select for Province
        if 'Province' in filtered_df.columns and filtered_df['Province'].nunique() > 1:
            selected_provinces = st.sidebar.multiselect(
                "Filter by Province",
                options=filtered_df['Province'].unique().tolist(),
                default=filtered_df['Province'].unique().tolist()
            )
            if selected_provinces:
                filtered_df = filtered_df[filtered_df['Province'].isin(selected_provinces)]
        elif 'Province' in filtered_df.columns and filtered_df['Province'].nunique() <= 1 and 'N/A' in filtered_df['Province'].unique():
            st.sidebar.info("Province filter not available: No province data or only 'N/A' found.")

        # Multi-select for Town (NEW Filter)
        if 'Town' in filtered_df.columns and filtered_df['Town'].nunique() > 1:
            selected_towns = st.sidebar.multiselect(
                "Filter by Town",
                options=filtered_df['Town'].unique().tolist(),
                default=filtered_df['Town'].unique().tolist()
            )
            if selected_towns:
                filtered_df = filtered_df[filtered_df['Town'].isin(selected_towns)]
        elif 'Town' in filtered_df.columns and filtered_df['Town'].nunique() <= 1 and 'N/A' in filtered_df['Town'].unique():
            st.sidebar.info("Town filter not available: No town data or only 'N/A' found.")

        # Multi-select for Vendor (NEW Filter)
        if 'Vendor' in filtered_df.columns and filtered_df['Vendor'].nunique() > 1:
            selected_vendors = st.sidebar.multiselect(
                "Filter by Vendor",
                options=filtered_df['Vendor'].unique().tolist(),
                default=filtered_df['Vendor'].unique().tolist()
            )
            if selected_vendors:
                filtered_df = filtered_df[filtered_df['Vendor'].isin(selected_vendors)]
        elif 'Vendor' in filtered_df.columns and filtered_df['Vendor'].nunique() <= 1 and 'N/A' in filtered_df['Vendor'].unique():
            st.sidebar.info("Vendor filter not available: No vendor data or only 'N/A' found.")


        # Multi-select for Bandwidth
        selected_bandwidths = st.sidebar.multiselect(
            "Filter by Bandwidth",
            options=filtered_df['BANDWIDTH'].unique().tolist(),
            default=filtered_df['BANDWIDTH'].unique().tolist()
        )
        if selected_bandwidths:
            filtered_df = filtered_df[filtered_df['BANDWIDTH'].isin(selected_bandwidths)]

        # Multi-select for Cell Level Congestion (renamed)
        selected_colors = st.sidebar.multiselect(
            "Filter by Cell Level Congestion", # Updated label
            options=filtered_df['Cell Level Congestion'].unique().tolist(), # Updated column name
            default=filtered_df['Cell Level Congestion'].unique().tolist() # Updated column name
        )
        if selected_colors:
            filtered_df = filtered_df[filtered_df['Cell Level Congestion'].isin(selected_colors)] # Updated column name

        # Multi-select for Final Assessment
        selected_assessments = st.sidebar.multiselect(
            "Filter by Final Assessment",
            options=filtered_df['FINAL ASSESSMENT'].unique().tolist(),
            default=filtered_df['FINAL ASSESSMENT'].unique().tolist()
        )
        if selected_assessments:
            filtered_df = filtered_df[filtered_df['FINAL ASSESSMENT'].isin(selected_assessments)]

        # Multi-select for Sector Level Congestion (renamed)
        selected_ctsc_colors = st.sidebar.multiselect(
            "Filter by Sector Level Congestion", # Updated label
            options=filtered_df['Sector Level Congestion'].unique().tolist(), # Updated column name
            default=filtered_df['Sector Level Congestion'].unique().tolist() # Updated column name
        )
        if selected_ctsc_colors:
            filtered_df = filtered_df[filtered_df['Sector Level Congestion'].isin(selected_ctsc_colors)] # Updated column name

        # Display the filtered DataFrame
        st.dataframe(filtered_df.drop_duplicates(subset=['Cell Name']), use_container_width=True)

        # Export to CSV button
        csv_export = filtered_df.drop_duplicates(subset=['Cell Name']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Unique Cells as CSV",
            data=csv_export,
            file_name="congested_unique_cells_analysis.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        # Cell Level Summary Metrics (based on unique cells)
        total_unique_cells = final_results_df['Cell Name'].nunique()
        red_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'RED'].shape[0]
        amber_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'AMBER'].shape[0]
        yellow_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'YELLOW'].shape[0]
        green_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'GREEN'].shape[0]
        blue_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'BLUE'].shape[0]
        black_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'BLACK'].shape[0]
        unknown_cell_color_cells = final_results_df[final_results_df['Cell Level Congestion'] == 'UNKNOWN_COLOR'].shape[0]
        
        # Final Assessment Metrics (based on unique cells for these counts)
        unique_final_assessment_df = final_results_df.drop_duplicates(subset=['Cell Name'])
        for_traffic_balancing_cells = unique_final_assessment_df[unique_final_assessment_df['FINAL ASSESSMENT'] == 'FOR TRAFFIC BALANCING'].shape[0]
        need_expansion_cells = unique_final_assessment_df[unique_final_assessment_df['FINAL ASSESSMENT'] == 'NEED EXPANSION'].shape[0]


        col1.metric("Total Unique Cells Analyzed", total_unique_cells)
        col2.metric("Red Cells", red_cells)
        col3.metric("Amber Cells", amber_cells)
        col1.metric("Yellow Cells", yellow_cells)
        col2.metric("Green Cells", green_cells)
        col3.metric("Blue Cells", blue_cells)
        col1.metric("Black Cells", black_cells)
        col2.metric("Unknown Cell Colors", unknown_cell_color_cells)
        col3.metric("Cells For Traffic Balancing", for_traffic_balancing_cells) 
        col1.metric("Cells Needing Expansion", need_expansion_cells)

        st.markdown("---")
        st.subheader("Sector-Level Summary (Unique Sectors)")

        # Calculate Sector Level Congestion counts based on unique sectors
        sector_summary_unique = final_results_df.drop_duplicates(subset=['SECTORNAME']) 
        
        sector_red_count = sector_summary_unique[sector_summary_unique['Sector Level Congestion'] == 'RED']['SECTORNAME'].nunique()
        sector_amber_count = sector_summary_unique[sector_summary_unique['Sector Level Congestion'] == 'AMBER']['SECTORNAME'].nunique()
        sector_green_count = sector_summary_unique[sector_summary_unique['Sector Level Congestion'] == 'GREEN']['SECTORNAME'].nunique()
        sector_black_count = sector_summary_unique[sector_summary_unique['Sector Level Congestion'] == 'BLACK']['SECTORNAME'].nunique()
        sector_unknown_color_count = sector_summary_unique[sector_summary_unique['Sector Level Congestion'] == 'UNKNOWN_COLOR']['SECTORNAME'].nunique()
        total_unique_sectors = sector_summary_unique['SECTORNAME'].nunique()

        col4.metric("Total Unique Sectors", total_unique_sectors)
        col5.metric("Red Sectors", sector_red_count)
        col6.metric("Amber Sectors", sector_amber_count)
        col4.metric("Green Sectors", sector_green_count)
        col5.metric("Black Sectors", sector_black_count)
        col6.metric("Unknown Sector Colors", sector_unknown_color_count)

        st.markdown("---")
        st.subheader("Visualizations")

        # Define color map for consistency and better visual representation for Altair
        alt_color_map = {
            'RED': 'red',
            'AMBER': 'orange',
            'YELLOW': 'yellow',
            'GREEN': 'green',
            'BLUE': 'blue',
            'BLACK': 'black',
            'UNKNOWN_COLOR': 'grey'
        }
        color_domain = list(alt_color_map.keys())
        color_range = list(alt_color_map.values())


        # Chart 1: Distribution of Cell Color Codes (Updated column name)
        st.markdown("#### Distribution of Cell Level Congestion") 
        color_counts = final_results_df.drop_duplicates(subset=['Cell Name'])['Cell Level Congestion'].value_counts().reset_index()
        color_counts.columns = ['Cell Level Congestion', 'Count']
        fig_colors = alt.Chart(color_counts).mark_bar().encode(
            x=alt.X('Cell Level Congestion', sort=color_domain, title='Congestion Level'),
            y=alt.Y('Count', title='Number of Cells'),
            color=alt.Color('Cell Level Congestion', scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title="Congestion Level"))
        ).properties(
            title='Overall Cell Level Congestion Distribution (Unique Cells)'
        ).interactive()
        st.altair_chart(fig_colors, use_container_width=True)

        # Chart 2: Distribution of Final Assessment (No change in column names)
        st.markdown("#### Distribution of Final Assessment Status")
        assessment_counts = final_results_df.drop_duplicates(subset=['Cell Name'])['FINAL ASSESSMENT'].value_counts().reset_index()
        assessment_counts.columns = ['Assessment Status', 'Count']
        fig_assessment = alt.Chart(assessment_counts).mark_bar().encode(
            x=alt.X('Assessment Status', title='Assessment'),
            y=alt.Y('Count', title='Number of Cells'),
            color=alt.Color('Assessment Status', legend=alt.Legend(title="Assessment Status"))
        ).properties(
            title='Overall Final Assessment Distribution (Unique Cells)'
        ).interactive()
        st.altair_chart(fig_assessment, use_container_width=True)

        # Chart 3: Average PRB and Traffic by Bandwidth (No change in column names)
        st.markdown("#### Average PRB and Traffic by Bandwidth")
        avg_kpis_by_bandwidth = final_results_df.groupby('BANDWIDTH')[['Final PRB', 'Final Traffic']].mean().reset_index()
        melted_kpis = avg_kpis_by_bandwidth.melt(id_vars='BANDWIDTH', var_name='Metric', value_name='Average Value')
        fig_kpi_bandwidth = alt.Chart(melted_kpis).mark_bar().encode(
            x=alt.X('BANDWIDTH', title='Bandwidth'),
            y=alt.Y('Average Value', title='Average KPI Value'),
            color=alt.Color('Metric', legend=alt.Legend(title="Key Performance Indicator")),
            column=alt.Column('Metric', header=alt.Header(titleOrient="bottom", labelOrient="bottom"))
        ).properties(
            title='Average Final PRB and Final Traffic per Bandwidth'
        ).interactive()
        st.altair_chart(fig_kpi_bandwidth, use_container_width=True)

        # Chart 4: Scatter plot of Final PRB vs Final Traffic, colored by Cell Level Congestion (Updated column name)
        st.markdown("#### Final PRB vs Final Traffic by Cell Level Congestion")
        fig_scatter = alt.Chart(final_results_df.drop_duplicates(subset=['Cell Name']).dropna(subset=['Final PRB', 'Final Traffic', 'Cell Level Congestion'])).mark_point().encode(
            x=alt.X('Final PRB', title='Final PRB Usage'),
            y=alt.Y('Final Traffic', title='Final User Traffic'),
            color=alt.Color('Cell Level Congestion', scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title="Congestion Level")),
            tooltip=['Cell Name', 'Final PRB', 'Final Traffic', 'Cell Level Congestion']
        ).properties(
            title='PRB vs. Traffic per Cell, Colored by Congestion Status (Unique Cells)'
        ).interactive()
        st.altair_chart(fig_scatter, use_container_width=True)

        # Chart 5: Sector Level Congestion Distribution (Updated column name)
        st.markdown("#### Sector Level Congestion Distribution")
        ctsc_counts = final_results_df.drop_duplicates(subset=['Cell Name'])['Sector Level Congestion'].value_counts().reset_index()
        ctsc_counts.columns = ['Sector Level Congestion', 'Count']
        ctsc_color_map = {
            'RED': 'red',
            'AMBER': 'orange',
            'GREEN': 'green',
            'BLACK': 'black',
            'UNKNOWN_COLOR': 'grey'
        }
        ctsc_color_domain = list(ctsc_color_map.keys())
        ctsc_color_range = list(ctsc_color_map.values())

        fig_ctsc = alt.Chart(ctsc_counts).mark_bar().encode(
            x=alt.X('Sector Level Congestion', sort=ctsc_color_domain, title='Sector Level Status'),
            y=alt.Y('Count', title='Number of Sectors/Cells'),
            color=alt.Color('Sector Level Congestion', scale=alt.Scale(domain=ctsc_color_domain, range=ctsc_color_range),
                            legend=alt.Legend(title="Sector Level Status"))
        ).properties(
            title='Distribution of Sector Level Congestion (Unique Cells)'
        ).interactive()
        st.altair_chart(fig_ctsc, use_container_width=True)

        # Chart for Cell Status by Province (Grouped Bar Chart) (Updated column name)
        if 'Province' in final_results_df.columns and final_results_df['Province'].nunique() > 1:
            st.markdown("#### Cell Status by Province")
            province_color_counts = final_results_df.drop_duplicates(subset=['Cell Name']).groupby(['Province', 'Cell Level Congestion']).size().reset_index(name='Count')
            
            fig_province_status = alt.Chart(province_color_counts).mark_bar().encode(
                x=alt.X('Count', title='Number of Cells'),
                y=alt.Y('Province', sort='-x', title='Province'),
                color=alt.Color('Cell Level Congestion', scale=alt.Scale(domain=color_domain, range=color_range),
                                legend=alt.Legend(title="Congestion Level")),
                order=alt.Order('Cell Level Congestion', sort='descending')
            ).properties(
                title='Cell Congestion Status per Province (Unique Cells)'
            ).interactive()
            st.altair_chart(fig_province_status, use_container_width=True)
        elif 'Province' in final_results_df.columns and final_results_df['Province'].nunique() <= 1:
            st.markdown("#### Cell Status by Province")
            st.info("No sufficient 'Province' data to display this chart (only one or no unique province found).")

        # Chart: Cell Status by Town (Grouped Bar Chart)
        if 'Town' in final_results_df.columns and final_results_df['Town'].nunique() > 1:
            st.markdown("#### Cell Status by Town")
            town_color_counts = final_results_df.drop_duplicates(subset=['Cell Name']).groupby(['Town', 'Cell Level Congestion']).size().reset_index(name='Count')
            
            fig_town_status = alt.Chart(town_color_counts).mark_bar().encode(
                x=alt.X('Count', title='Number of Cells'),
                y=alt.Y('Town', sort='-x', title='Town'),
                color=alt.Color('Cell Level Congestion', scale=alt.Scale(domain=color_domain, range=color_range),
                                legend=alt.Legend(title="Congestion")),
                order=alt.Order('Cell Level Congestion', sort='descending')
            ).properties(
                title='Cell Congestion Status per Town (Unique Cells)'
            ).interactive()
            st.altair_chart(fig_town_status, use_container_width=True)
        elif 'Town' in final_results_df.columns and final_results_df['Town'].nunique() <= 1:
            st.markdown("#### Cell Status by Town")
            st.info("No sufficient 'Town' data to display this chart (only one or no unique town found).")

        # NEW Chart: Cell Level Congestion per Vendor
        if 'Vendor' in final_results_df.columns and final_results_df['Vendor'].nunique() > 1:
            st.markdown("#### Cell Level Congestion per Vendor")
            vendor_color_counts = final_results_df.drop_duplicates(subset=['Cell Name']).groupby(['Vendor', 'Cell Level Congestion']).size().reset_index(name='Count')
            
            fig_vendor_status = alt.Chart(vendor_color_counts).mark_bar().encode(
                x=alt.X('Count', title='Number of Cells'),
                y=alt.Y('Vendor', sort='-x', title='Vendor'),
                color=alt.Color('Cell Level Congestion', scale=alt.Scale(domain=color_domain, range=color_range),
                                legend=alt.Legend(title="Congestion Level")),
                order=alt.Order('Cell Level Congestion', sort='descending')
            ).properties(
                title='Cell Congestion Status per Vendor (Unique Cells)'
            ).interactive()
            st.altair_chart(fig_vendor_status, use_container_width=True)
        elif 'Vendor' in final_results_df.columns and final_results_df['Vendor'].nunique() <= 1:
            st.markdown("#### Cell Level Congestion per Vendor")
            st.info("No sufficient 'Vendor' data to display this chart (only one or no unique vendor found).")


        st.markdown("---")
        st.info("Analysis complete. You can filter the results using the sidebar options.")
    except Exception as e:
        st.error(f"Error processing data: {e}. Please check your CSV file format and data integrity.")
else:
    st.info("Please upload a CSV file to proceed with the analysis.")
