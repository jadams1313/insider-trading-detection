import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load GKG File with Proper Tab Delimiter
# ============================================================================

def load_gkg_for_insider_trading(filepath):
    """
    Load GKG file and keep only columns relevant for insider trading detection
    
    Kept columns:
    - GKGRECORDID: Unique identifier
    - DATE: When article was published
    - SourceCommonName: News source (domain)
    - DocumentIdentifier: URL to article
    - Themes: Topic/theme codes
    - Organizations: Companies mentioned
    - Persons: People mentioned
    - V2Tone: Sentiment metrics
    - GCAM: Emotion/theme scores
    - Quotations: Quotes in article
    - AllNames: All entities mentioned
    """
    
    # Full GKG 2.1 column names
    all_gkg_columns = [
        'GKGRECORDID',              # 0  - KEEP: Unique ID
        'DATE',                      # 1  - KEEP: Publication date
        'SourceCollectionIdentifier',# 2  - DROP: Just says if it's web/print
        'SourceCommonName',          # 3  - KEEP: News source domain
        'DocumentIdentifier',        # 4  - KEEP: URL to article
        'Counts',                    # 5  - DROP: Legacy field
        'V2Counts',                  # 6  - DROP: Complex entity counts
        'Themes',                    # 7  - KEEP: Topic codes (simplified)
        'V2Themes',                  # 8  - DROP: Detailed themes (too complex)
        'Locations',                 # 9  - DROP: Geographic mentions
        'V2Locations',               # 10 - DROP: Detailed locations
        'Persons',                   # 11 - KEEP: People mentioned
        'V2Persons',                 # 12 - DROP: Detailed person info
        'Organizations',             # 13 - KEEP: Companies mentioned
        'V2Organizations',           # 14 - DROP: Detailed org info
        'V2Tone',                    # 15 - KEEP: Sentiment scores
        'Dates',                     # 16 - DROP: Dates mentioned in text
        'GCAM',                      # 17 - KEEP: Emotion/theme dimensions
        'SharingImage',              # 18 - DROP: Social media images
        'RelatedImages',             # 19 - DROP: Article images
        'SocialImageEmbeds',         # 20 - DROP: Embedded images
        'SocialVideoEmbeds',         # 21 - DROP: Embedded videos
        'Quotations',                # 22 - KEEP: Direct quotes
        'AllNames',                  # 23 - KEEP: All entity names
        'Amounts',                   # 24 - DROP: Numbers mentioned
        'TranslationInfo',           # 25 - DROP: Translation metadata
        'Extras',                    # 26 - DROP: Reserved field
    ]
    
    # Columns to keep for insider trading analysis
    keep_columns = [
        'GKGRECORDID',        # ID
        'DATE',               # When
        'SourceCommonName',   # Where (news source)
        'DocumentIdentifier', # Link to article
        'Themes',             # What topics
        'Persons',            # Who (executives, traders)
        'Organizations',      # Which companies
        'V2Tone',             # Sentiment
        'GCAM',               # Detailed emotions/themes
        'Quotations',         # Direct quotes
        'AllNames',           # All entities
    ]
    
    try:
        # Load with all columns first
        df = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=all_gkg_columns,
            low_memory=False,
            on_bad_lines='skip',
            encoding='utf-8',
            quoting=3,  # QUOTE_NONE
            dtype=str   # Read everything as string first
        )
                
        # Keep only relevant columns
        df = df[keep_columns]        
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None



#Parse and Clean Key Fields
def parse_gkg_fields(df):
    """
    Parse complex fields into usable format
    """
    
    # Parse V2Tone (comma-separated: tone,positive,negative,polarity,activity,wordcount)
    if 'V2Tone' in df.columns and df['V2Tone'].notna().any():
        tone_parts = df['V2Tone'].str.split(',', expand=True)
        df['Tone'] = pd.to_numeric(tone_parts[0], errors='coerce')
        df['PositiveScore'] = pd.to_numeric(tone_parts[1], errors='coerce')
        df['NegativeScore'] = pd.to_numeric(tone_parts[2], errors='coerce')
        df['Polarity'] = pd.to_numeric(tone_parts[3], errors='coerce')
        df['ActivityRefDensity'] = pd.to_numeric(tone_parts[4], errors='coerce')
        df['WordCount'] = pd.to_numeric(tone_parts[5], errors='coerce')
    
    # Convert DATE to datetime
    if 'DATE' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S', errors='coerce')
        df['Date'] = df['DateTime'].dt.date
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['Day'] = df['DateTime'].dt.day
        df['Hour'] = df['DateTime'].dt.hour
    
    # Clean Organizations (semicolon-separated)
    if 'Organizations' in df.columns:
        df['OrgCount'] = df['Organizations'].str.count(';') + 1
        df['OrgCount'] = df['OrgCount'].fillna(0).astype(int)
        # Extract first organization for quick filtering
        df['PrimaryOrg'] = df['Organizations'].str.split(';').str[0]
    
    # Clean Persons (semicolon-separated)
    if 'Persons' in df.columns:
        df['PersonCount'] = df['Persons'].str.count(';') + 1
        df['PersonCount'] = df['PersonCount'].fillna(0).astype(int)
        # Extract first person
        df['PrimaryPerson'] = df['Persons'].str.split(';').str[0]
    
    # Extract theme keywords (simplified)
    if 'Themes' in df.columns and df['Themes'].notna().any():
        # Check if article mentions financial/business themes
        df['HasFinancialTheme'] = df['Themes'].str.contains(
            'ECON|BUSINESS|FINANCE|MARKET|TRADE|TAX',
            case=False,
            na=False
        )
        
    
    return df


#Filter for Insider Trading Relevant Articles

def filter_for_insider_trading(df):
    """
    Filter articles likely relevant to insider trading detection
    """
    print("\nFiltering for insider trading relevance...")
    
    # Keywords that suggest PERM-R events or insider trading
    insider_keywords = [
        'insider', 'trading', 'sec filing', 'form 4', 'form 3', 
        'earnings', 'merger', 'acquisition', 'fda approval',
        'product launch', 'clinical trial', 'regulatory',
        'executive', 'ceo', 'cfo', 'director', 'officer',
        'stock sale', 'stock purchase', 'shares sold', 'shares bought'
    ]
    
    # Create pattern for regex search
    pattern = '|'.join(insider_keywords)
    
    # Filter based on themes, quotations, or document content
    mask = (
        df['Themes'].str.contains(pattern, case=False, na=False) |
        df['Quotations'].str.contains(pattern, case=False, na=False) |
        df['DocumentIdentifier'].str.contains(pattern, case=False, na=False)
    )
    
    filtered_df = df[mask].copy()
    
    print(f"  ✓ Found {len(filtered_df):,} potentially relevant articles")
    print(f"  ✓ Filtered from {len(df):,} total articles ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df


# ============================================================================
# STEP 4: Company Mapping and Filtering
# ============================================================================

def create_company_mapping():
    """
    Map ticker symbols to company names for filtering
    """
    company_map = {
        "AMSC": ["American Superconductor", "AMSC"],
        "NP": ["Neenah", "Neenah Paper", "Neenah Inc"],
        "EVR": ["Evercore"],
        "GOOGL": ["Google", "Alphabet", "GOOGL", "GOOG"],
        "GTXI": ["GTx", "GTx Inc"],
        "HLF": ["Herbalife"],
        "MDRX": ["Veradigm", "Allscripts", "MDRX"],
        "ORCL": ["Oracle", "Oracle Corporation"],
        "SPPI": ["Spectrum Pharmaceuticals", "SPPI"],
        "WFC": ["Wells Fargo", "Wells Fargo Bank", "WFC"]
    }
    return company_map


def filter_by_companies(df, tickers):
    """
    Filter GKG data for specific companies
    
    Parameters:
    - df: GKG DataFrame
    - tickers: List of ticker symbols
    
    Returns:
    - Filtered DataFrame with company column added
    """

    company_map = create_company_mapping()
    
    # Build search patterns for each company
    all_matches = []
    company_stats = {}
    
    for ticker in tickers:
        company_names = company_map[ticker]
        
        # Create case-insensitive regex pattern
        # Add word boundaries to avoid partial matches
        pattern = '|'.join([f'\\b{name}\\b' for name in company_names])
        
        # Search in Organizations, AllNames, and DocumentIdentifier
        mask = (
            df['Organizations'].str.contains(pattern, case=False, na=False, regex=True) |
            df['AllNames'].str.contains(pattern, case=False, na=False, regex=True) |
            df['DocumentIdentifier'].str.contains(pattern, case=False, na=False, regex=True)
        )
        
        company_articles = df[mask].copy()
        company_articles['Ticker'] = ticker
        company_articles['CompanyName'] = company_names[0]  # Primary name
        
        all_matches.append(company_articles)
        company_stats[ticker] = len(company_articles)
        
        print(f"  {ticker:6} ({company_names[0]:30}): {len(company_articles):5,} articles")
    
    if not all_matches:
        return pd.DataFrame()
    
    # Combine all company matches
    df_companies = pd.concat(all_matches, ignore_index=True)
    
    # Remove duplicates (articles mentioning multiple companies)
    # Keep first occurrence
    df_companies = df_companies.drop_duplicates(subset=['GKGRECORDID'], keep='first')

    return df_companies


def analyze_company_coverage(df_companies):
    
    tone_by_company = df_companies.groupby('Ticker')['Tone'].agg(['mean', 'std', 'count'])
    tone_by_company = tone_by_company.sort_values('mean', ascending=False)
    
    coverage_by_date = df_companies.groupby('Date').size().sort_index()
    
    most_polar = df_companies.nlargest(10, 'Polarity')[
        ['Date', 'Ticker', 'CompanyName', 'Tone', 'Polarity', 'SourceCommonName']
    ]
    print(most_polar.to_string(index=False))
    
    return df_companies


def create_company_timeseries(df_companies):
    """
    Create time series of coverage and sentiment for each company
    """
    print("\n" + "="*60)
    print("TIME SERIES DATA")
    print("="*60)
    
    # Daily coverage count and average tone per company
    timeseries = df_companies.groupby(['Date', 'Ticker']).agg({
        'GKGRECORDID': 'count',  # Article count
        'Tone': 'mean',
        'Polarity': 'mean',
        'WordCount': 'mean'
    }).rename(columns={'GKGRECORDID': 'ArticleCount'})
    
    timeseries = timeseries.reset_index()
    
    print(f"✓ Created time series with {len(timeseries)} date-company pairs")
    print("\nSample:")
    print(timeseries.head(10))
    
    return timeseries


#target companies
tickers = [
    "AMSC",   # American Superconductor
    "NP",     # Neenah Paper / Neenah Inc.
    "EVR",    # Evercore
    "GOOGL",  # Google (Alphabet Class A)
    "GTXI",   # GTx Inc.
    "HLF",    # Herbalife
    "MDRX",   # Veradigm (formerly Allscripts)
    "ORCL",   # Oracle
    "SPPI",   # Spectrum Pharmaceuticals
    "WFC"     # Wells Fargo
]
