import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
import altair as alt
import os
import plotly.graph_objects as go

st.markdown(
    '''
    <style>
    /* Reduce the size of multiselect tags */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 0.8rem !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        height: 1.5em !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Load the data
csv_path = r"2025jan-jul.csv"
df = pd.read_csv(csv_path, encoding='latin1')

# Clean numeric columns (remove commas, handle missing values)
def clean_numeric(col):
    return pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')

# Columns of interest
outstanding_col = 'Outstanding'
principal_col = 'Principal Amount'
interest_col = 'Interest Amount'

# Clean columns
for col in [outstanding_col, principal_col, interest_col]:
    if col in df.columns:
        df[col] = clean_numeric(col)

# Ensure Disbursed Date is parsed early for month filtering
if 'Disbursed Date' in df.columns:
    df['Disbursed Date'] = pd.to_datetime(df['Disbursed Date'], errors='coerce', dayfirst=True)
    df['Month'] = df['Disbursed Date'].dt.strftime('%b %Y')
else:
    df['Month'] = 'Unknown'

# Branch code to name mapping
branch_map = {
    '8550': 'CBD',
    '59535': 'ZIDISHA',
    '12936': 'BURUBURU',
    '27133': 'KILIMANI',
    '63796': 'KIAMBU',
    '55886': 'UTAWALA',
    '75350': 'THIKA',
}

branch_col = 'Branch'
if branch_col in df.columns:
    # Map codes to names, use 'Other' for unmapped
    df['Branch Name'] = df[branch_col].astype(str).map(branch_map).fillna('Other')
else:
    df['Branch Name'] = 'Other'

# Branch filter (already robust)
branch_names = sorted(df['Branch Name'].unique())

# Month filter
month_names = sorted(df['Month'].dropna().unique(), key=lambda x: pd.to_datetime(x, format='%b %Y', errors='coerce'))

# Layout: Title left, filters right, all on one row
header_col, branch_col_ui, month_col_ui = st.columns([2, 1, 1])
with header_col:
    st.markdown("<h1 style='margin-bottom:0;'>Loan Portfolio Dashboard</h1>", unsafe_allow_html=True)
with branch_col_ui:
    selected_branch = st.selectbox('Branch', options=['All'] + branch_names, key='branch_filter')
with month_col_ui:
    selected_month = st.selectbox('Month', options=['All'] + month_names, key='month_filter')

# Apply both filters
filtered_df = df
if selected_branch != 'All':
    filtered_df = filtered_df[filtered_df['Branch Name'] == selected_branch]
if selected_month != 'All':
    filtered_df = filtered_df[filtered_df['Month'] == selected_month]

# Update metrics and charts to use filtered_df
# 1. Total Outstanding Amount
if outstanding_col in filtered_df.columns:
    total_outstanding = filtered_df[outstanding_col].sum()
else:
    total_outstanding = None

# 2. Total Principal vs. Total Interest balance
if principal_col in filtered_df.columns:
    total_principal = filtered_df[principal_col].sum()
else:
    total_principal = None
if interest_col in filtered_df.columns:
    total_interest = filtered_df[interest_col].sum()
else:
    total_interest = None

# 3. Average loan balance (using Outstanding as proxy)
if outstanding_col in filtered_df.columns:
    avg_loan_balance = filtered_df[outstanding_col].mean()
else:
    avg_loan_balance = None

# Streamlit Dashboard
# Remove the 'Key Metrics' header
# Key Metrics in one row, with increased spacing between cards and right padding
col1, pad1, col2, pad2, col3, pad3, col4, col_pad = st.columns([1, 0.12, 1, 0.12, 1, 0.12, 1, 0.15])
col1.metric("Total Outstanding", f"{total_outstanding:,.2f}" if total_outstanding is not None else "N/A")
col2.metric("Total Principal", f"{total_principal:,.2f}" if total_principal is not None else "N/A")
col3.metric("Total Interest", f"{total_interest:,.2f}" if total_interest is not None else "N/A")
col4.metric("Average Loan Balance", f"{avg_loan_balance:,.2f}" if avg_loan_balance is not None else "N/A")



# --- Collections Performance Metrics ---
# Clean relevant columns
for col in ['Paid Amount', 'Total Due Amount', 'Penalty Amount', 'Pending Penalty Due', 'Days Past Due']:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')

# 1. Collection Ratio
if 'Paid Amount' in filtered_df.columns and 'Total Due Amount' in filtered_df.columns:
    total_paid = filtered_df['Paid Amount'].sum()
    total_due = filtered_df['Total Due Amount'].sum()
    collection_ratio = total_paid / total_due if total_due > 0 else None
else:
    collection_ratio = None

# 2. Recovery Rate (on overdue loans)
overdue_statuses = ['Arrears', 'Missed Repayment', 'Past Maturity', 'Past Due']
overdue_mask = (
    (filtered_df['Days Past Due'].fillna(0) > 0) |
    (filtered_df['Status'].isin(overdue_statuses) if 'Status' in filtered_df.columns else False)
)
overdue_df = filtered_df[overdue_mask]
if 'Paid Amount' in overdue_df.columns and 'Total Due Amount' in overdue_df.columns:
    paid_overdue = overdue_df['Paid Amount'].sum()
    overdue_due = overdue_df['Total Due Amount'].sum()
    recovery_rate = paid_overdue / overdue_due if overdue_due > 0 else None
else:
    recovery_rate = None

# 3. Penalty Income vs. Outstanding
if 'Penalty Amount' in filtered_df.columns and 'Pending Penalty Due' in filtered_df.columns:
    penalty_income = filtered_df['Penalty Amount'].sum()
    penalty_outstanding = filtered_df['Pending Penalty Due'].sum()
else:
    penalty_income = None
    penalty_outstanding = None

# Display metrics in a row
st.markdown("<h3 style='margin-top:2rem;margin-bottom:0.5rem;'>Collections Performance</h3>", unsafe_allow_html=True)
perf1, perf2, perf3, perf_pad = st.columns([1, 1, 1, 0.5])
perf1.metric("Collection Ratio", f"{collection_ratio:.2%}" if collection_ratio is not None else "N/A")
perf2.metric("Recovery Rate", f"{recovery_rate:.2%}" if recovery_rate is not None else "N/A")
perf3.metric("Penalty Income / Outstanding", f"{penalty_income:,.0f} / {penalty_outstanding:,.0f}" if penalty_income is not None and penalty_outstanding is not None else "N/A")

# --- Collections Performance Over Time Plot ---
# By month, show collection ratio, recovery rate, penalty income/outstanding
if 'Month' in filtered_df.columns:
    perf_month = filtered_df.copy()
    # Clean columns again for groupby
    for col in ['Paid Amount', 'Total Due Amount', 'Penalty Amount', 'Pending Penalty Due', 'Days Past Due']:
        if col in perf_month.columns:
            perf_month[col] = pd.to_numeric(perf_month[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
    # Group by month
    def safe_div(num, denom):
        return num / denom if denom > 0 else None
    grouped = perf_month.groupby('Month').agg({
        'Paid Amount': 'sum',
        'Total Due Amount': 'sum',
        'Penalty Amount': 'sum',
        'Pending Penalty Due': 'sum',
        'Days Past Due': 'max',
        'Status': 'first'
    }).reset_index()
    # Collection Ratio
    grouped['Collection Ratio'] = grouped.apply(lambda r: safe_div(r['Paid Amount'], r['Total Due Amount']), axis=1)
    # Recovery Rate (on overdue)
    overdue_mask = (
        (perf_month['Days Past Due'].fillna(0) > 0) |
        (perf_month['Status'].isin(overdue_statuses) if 'Status' in perf_month.columns else False)
    )
    overdue_by_month = perf_month[overdue_mask].groupby('Month').agg({'Paid Amount': 'sum', 'Total Due Amount': 'sum'}).reset_index()
    grouped = pd.merge(grouped, overdue_by_month, on='Month', how='left', suffixes=('', '_Overdue'))
    grouped['Recovery Rate'] = grouped.apply(lambda r: safe_div(r.get('Paid Amount_Overdue', 0), r.get('Total Due Amount_Overdue', 0)), axis=1)
    # Penalty Income vs Outstanding
    grouped['Penalty Income'] = grouped['Penalty Amount']
    grouped['Penalty Outstanding'] = grouped['Pending Penalty Due']
    # Melt for Altair
    plot_df = grouped.melt(id_vars=['Month'], value_vars=['Collection Ratio', 'Recovery Rate'], var_name='Metric', value_name='Value')
    chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X('Month:N', title='Month', sort=month_names),
        y=alt.Y('Value:Q', title='Ratio', axis=alt.Axis(format='%')),
        color=alt.Color('Metric:N', title='Metric'),
        tooltip=['Month', 'Metric', alt.Tooltip('Value', format='.2%')]
    ).properties(title='Collection Ratio & Recovery Rate Over Time')
    # Penalty bar chart
    penalty_df = grouped[['Month', 'Penalty Income', 'Penalty Outstanding']].melt(id_vars=['Month'], var_name='Type', value_name='Amount')
    penalty_chart = alt.Chart(penalty_df).mark_bar().encode(
        x=alt.X('Month:N', title='Month', sort=month_names),
        y=alt.Y('Amount:Q', title='Amount'),
        color=alt.Color('Type:N', title='Type'),
        tooltip=['Month', 'Type', alt.Tooltip('Amount', format=',.0f')]
    ).properties(title='Penalty Income vs Outstanding Over Time')
    # Show plots side by side
    left_col, right_col = st.columns(2)
    with left_col:
        st.altair_chart(chart, use_container_width=True)
    with right_col:
        st.altair_chart(penalty_chart, use_container_width=True)

# --- Combined Disbursement and Collections Trends Over Time ---
# Only plot if both data sources are available
if 'Disbursed Date' in filtered_df.columns and 'Disbursed' in filtered_df.columns and os.path.exists('collections.csv'):
    # Prepare disbursement trend
    disb_trend = filtered_df.groupby('Disbursed Date')['Disbursed'].sum().sort_index().reset_index()
    disb_trend = disb_trend.rename(columns={'Disbursed Date': 'Date', 'Disbursed': 'Amount'})
    disb_trend['Type'] = 'Disbursement'
    # Prepare collections trend, filtered by branch and month
    collections_df = pd.read_csv('collections.csv', encoding='latin1')
    collections_df['Collection Date'] = pd.to_datetime(collections_df['Collection Date'], errors='coerce', dayfirst=True)
    collections_df['Total Paid Amount'] = pd.to_numeric(collections_df['Total Paid Amount'], errors='coerce')
    # Map branch codes to names for collections
    if 'Branch' in collections_df.columns:
        collections_df['Branch'] = collections_df['Branch'].astype(str).map(branch_map).fillna(collections_df['Branch'].astype(str))
    # Add month column for filtering
    collections_df['Month'] = collections_df['Collection Date'].dt.strftime('%b %Y')
    # Apply filters
    if selected_branch != 'All':
        collections_df = collections_df[collections_df['Branch'] == selected_branch]
    if selected_month != 'All':
        collections_df = collections_df[collections_df['Month'] == selected_month]
    collections_trend = collections_df.groupby('Collection Date')['Total Paid Amount'].sum().reset_index()
    collections_trend = collections_trend.rename(columns={'Collection Date': 'Date', 'Total Paid Amount': 'Amount'})
    collections_trend['Type'] = 'Collection'
    # Combine for Altair
    combined_trend = pd.concat([disb_trend, collections_trend], ignore_index=True)
    st.header('Disbursement and Collections Trends Over Time')
    trend_options = ['Both', 'Disbursement Only', 'Collections Only']
    selected_trend = st.radio('Show:', trend_options, horizontal=True)
    if selected_trend == 'Disbursement Only':
        plot_data = combined_trend[combined_trend['Type'] == 'Disbursement']
        base = alt.Chart(plot_data).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Amount:Q', title='Amount'),
            color=alt.value('#1f77b4'),
            tooltip=['Date', alt.Tooltip('Amount', format=',.2f')]
        )
        trend = base.transform_regression('Date', 'Amount').mark_line(color='red', strokeDash=[4,4])
        chart = base + trend
    elif selected_trend == 'Collections Only':
        plot_data = combined_trend[combined_trend['Type'] == 'Collection']
        base = alt.Chart(plot_data).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Amount:Q', title='Amount'),
            color=alt.value('#ff7f0e'),
            tooltip=['Date', alt.Tooltip('Amount', format=',.2f')]
        )
        trend = base.transform_regression('Date', 'Amount').mark_line(color='red', strokeDash=[4,4])
        chart = base + trend
    else:
        plot_data = combined_trend
        chart = alt.Chart(plot_data).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Amount:Q', title='Amount'),
            color=alt.Color('Type:N', title='Type'),
            tooltip=['Date', 'Type', alt.Tooltip('Amount', format=',.2f')]
        )
    st.altair_chart(chart, use_container_width=True)

# --- Loan Product Distribution and Status Distribution by Loan Product (side by side) ---
if 'Loan Product' in filtered_df.columns:
    prod_counts = filtered_df['Loan Product'].value_counts().reset_index()
    prod_counts.columns = ['Loan Product', 'Count']

left_col, right_col = st.columns([1, 1])
with left_col:
    if 'Loan Product' in filtered_df.columns:
        fig = go.Figure(data=[go.Pie(
            labels=prod_counts['Loan Product'],
            values=prod_counts['Count'],
            hole=0.4,
            pull=[0.05]*len(prod_counts),
            marker=dict(line=dict(color='#000000', width=1)),
            textinfo='label+percent',
            hoverinfo='label+value+percent',
        )])
        fig.update_traces(textfont_size=14)
        st.header('Loan Product Distribution (Number of Loans)')
        fig.update_layout(
            showlegend=True,
            legend=dict(font=dict(size=12)),
            height=600,
            width=900,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=False)
with right_col:
    if 'Loan Product' in filtered_df.columns and 'Status' in filtered_df.columns:
        status_prod_counts = filtered_df.groupby(['Loan Product', 'Status']).size().reset_index(name='Count')
        st.header('Status Distribution by Loan Product')
        chart = alt.Chart(status_prod_counts).mark_bar().encode(
            x=alt.X('Loan Product:N', title='Loan Product', sort=alt.EncodingSortField(field='Count', op='sum', order='descending')),
            y=alt.Y('Count:Q', title='Number of Loans'),
            color=alt.Color('Status:N', title='Status'),
            tooltip=['Loan Product', 'Status', 'Count']
        ).properties(
            width=700, height=500,
        )
        st.altair_chart(chart, use_container_width=False)

# --- PAR per branch over the months ---
# Define PAR statuses
par_statuses = ['Arrears', 'Missed Repayment', 'Past Maturity', 'Past Due']

if 'Outstanding' in df.columns and 'Status' in df.columns and 'Branch Name' in df.columns and 'Month' in df.columns:
    par_df = df.copy()
    # Clean Outstanding
    par_df['Outstanding'] = pd.to_numeric(par_df['Outstanding'], errors='coerce')
    # Calculate total and at-risk outstanding per branch and month
    total_out = par_df.groupby(['Branch Name', 'Month'])['Outstanding'].sum().reset_index(name='Total Outstanding')
    par_out = par_df[par_df['Status'].isin(par_statuses)].groupby(['Branch Name', 'Month'])['Outstanding'].sum().reset_index(name='PAR Outstanding')
    # Merge and calculate PAR%
    par_merged = pd.merge(total_out, par_out, on=['Branch Name', 'Month'], how='left').fillna(0)
    par_merged['PAR%'] = (par_merged['PAR Outstanding'] / par_merged['Total Outstanding']) * 100
    # Calculate overall PAR per month (all branches)
    overall_total = par_df.groupby('Month')['Outstanding'].sum().reset_index(name='Total Outstanding')
    overall_par = par_df[par_df['Status'].isin(par_statuses)].groupby('Month')['Outstanding'].sum().reset_index(name='PAR Outstanding')
    overall_merged = pd.merge(overall_total, overall_par, on='Month', how='left').fillna(0)
    overall_merged['PAR%'] = (overall_merged['PAR Outstanding'] / overall_merged['Total Outstanding']) * 100
    overall_merged['Branch Name'] = 'All Branches'
    # Combine with branch-level data
    par_final = pd.concat([par_merged, overall_merged], ignore_index=True)
    # Layout: PAR chart on left half, status-by-product on right half
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.header('PAR per Branch Over the Months')
        par_branch_options = ['All Branches'] + sorted([b for b in par_merged['Branch Name'].unique() if b != 'All Branches'])
        selected_par_branches = st.multiselect('Select Branches for PAR Chart', options=par_branch_options, default=par_branch_options)
        par_final_filtered = par_final[par_final['Branch Name'].isin(selected_par_branches)]
        chart = alt.Chart(par_final_filtered).mark_line(point=True).encode(
            x=alt.X('Month:N', title='Month', sort=month_names),
            y=alt.Y('PAR%:Q', title='PAR (%)'),
            color=alt.Color('Branch Name:N', title='Branch'),
            tooltip=['Branch Name', 'Month', alt.Tooltip('PAR%', format='.2f')]
        )
        st.altair_chart(chart, use_container_width=True)
    with right_col:
        # This block is now redundant as the pie chart is moved
        pass

# --- 3D Pie Chart for Loan Product Distribution ---
# This section is now redundant as the pie chart is moved

# --- Disbursements 2024 vs 2025 to July 15 ---
if os.path.exists('2024jan-jun.csv') and os.path.exists('2025jan-jul.csv'):
    # Branch code to name mapping
    branch_map = {
        '8550': 'CBD',
        '59535': 'ZIDISHA',
        '12936': 'BURUBURU',
        '27133': 'KILIMANI',
        '63796': 'KIAMBU',
        '75350': 'THIKA',
        '55886': 'UTAWALA'
    }
    def get_disbursement_df(path, year, date_start, date_end):
        df = pd.read_csv(path, dtype=str, index_col=False, encoding='latin1')
        df = df[df.iloc[:,0] != 'Disbursed Date']
        # Always create Branch column
        if df.columns[0] != 'Disbursed Date':
            split_cols = df.iloc[:,0].str.extract(r'^(\d{2}/\d{2}/\d{4})\s+(\d+)$')
            if not split_cols.isnull().all().all():
                df['Disbursed Date'] = split_cols[0]
                df['Branch'] = split_cols[1]
            else:
                df['Disbursed Date'] = df.iloc[:,0]
                if 'Branch' not in df.columns:
                    df['Branch'] = ''
        else:
            if 'Branch' not in df.columns:
                df['Branch'] = ''
        # Map branch codes to names
        df['Branch Name'] = df['Branch'].map(branch_map).fillna(df['Branch'])
        amt_col = [col for col in df.columns if col.strip().lower() == 'disbursed']
        if not amt_col:
            return pd.DataFrame()
        amt_col = amt_col[0]
        df['Disbursed Date'] = pd.to_datetime(df['Disbursed Date'].astype(str).str.strip(), errors='coerce', dayfirst=True)
        df[amt_col] = pd.to_numeric(df[amt_col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
        df = df.dropna(subset=['Disbursed Date', amt_col])
        df = df[(df['Disbursed Date'] >= date_start) & (df['Disbursed Date'] <= date_end)]
        out = df.groupby(['Disbursed Date', 'Branch Name'])[amt_col].sum().reset_index()
        out['Year'] = year
        out = out.rename(columns={'Disbursed Date': 'Disbursed Date', amt_col: 'Disbursed', 'Branch Name': 'Branch Name'})
        out = out.sort_values('Disbursed Date')
        return out
    disb_2024 = get_disbursement_df('2024jan-jun.csv', '2024', '2024-01-01', '2024-06-30')
    disb_2025 = get_disbursement_df('2025jan-jul.csv', '2025', '2025-01-01', '2025-07-15')
    def add_month_day(df):
        df['MonthDay'] = df['Disbursed Date'].dt.strftime('%m-%d')
        return df
    disb_2024 = add_month_day(disb_2024)
    disb_2025 = add_month_day(disb_2025)
    disb_compare = pd.concat([disb_2024, disb_2025], ignore_index=True)
    # Add month and branch filters (use branch names)
    month_name_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July']
    all_months = [m for m in month_name_order if m in disb_compare['Disbursed Date'].dt.strftime('%B').unique()]
    all_branches = sorted(disb_compare['Branch Name'].dropna().unique())
    # Layout: Header and filters on the same row, with smaller header font
    header_col, month_col, branch_col = st.columns([2, 1, 1])
    with header_col:
        st.markdown("""
            <h2 style='margin-bottom: 0.5rem; font-size: 2rem;'>Disbursements 2024 vs 2025 to July 15</h2>
        """, unsafe_allow_html=True)
    with month_col:
        selected_month = st.selectbox('Filter by Month', options=['All'] + all_months, key='disb_month_filter')
    with branch_col:
        selected_branch = st.selectbox('Filter by Branch', options=['All'] + all_branches, key='disb_branch_filter')
    filtered_compare = disb_compare.copy()
    if selected_month != 'All':
        filtered_compare = filtered_compare[filtered_compare['Disbursed Date'].dt.strftime('%B') == selected_month]
    if selected_branch != 'All':
        filtered_compare = filtered_compare[filtered_compare['Branch Name'] == selected_branch]
    # Ensure MonthDay is sorted in calendar order
    from datetime import datetime
    monthday_order = [datetime(2000, m, d).strftime('%m-%d') for m in range(1, 8) for d in range(1, 32)
                      if not (m == 2 and d > 29) and not (m in [4, 6] and d > 30)]
    # Only keep MonthDay values that are present in the data
    monthday_order = [md for md in monthday_order if md in filtered_compare['MonthDay'].unique()]
    chart = alt.Chart(filtered_compare).mark_line(point=True).encode(
        x=alt.X('MonthDay:N', title='Month-Day', sort=monthday_order),
        y=alt.Y('Disbursed:Q', title='Total Disbursed'),
        color=alt.Color('Year:N', title='Year'),
        tooltip=['MonthDay', 'Year', 'Branch Name', alt.Tooltip('Disbursed', format=',.2f')]
    )
    st.altair_chart(chart, use_container_width=True) 