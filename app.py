import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Forecasting Validation Dashboard", layout="wide")

# --- 1. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    excel_file = 'NorthPeak_Foods_Masked_Data_Pack.xlsx'
    csv_file = 'Monthly_Revenue_EBITDA.csv'
    
    def load_sheet_safely(filepath, sheet_name, keyword, usecols=None):
        df_temp = pd.read_excel(filepath, sheet_name=sheet_name, nrows=15, header=None)
        header_idx = 0
        for idx, row in df_temp.iterrows():
            if keyword in row.values:
                header_idx = idx
                break
        return pd.read_excel(filepath, sheet_name=sheet_name, header=header_idx, usecols=usecols)

    # 1A. Load Raw Monthly Data from Excel (Shows the Anomaly for Tab 1)
    try:
        df_monthly_excel = load_sheet_safely(excel_file, 'Monthly_Revenue_EBITDA', 'Year_Code')
    except FileNotFoundError:
        st.error(f"Error: Could not find '{excel_file}'. Please ensure it is in the same directory.")
        st.stop()

    # 1B. Load Cleaned Monthly Data from CSV (Used for Forecasting in Tab 3)
    try:
        df_monthly_csv = pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"Error: Could not find '{csv_file}'. Please ensure it is in the same directory.")
        st.stop()
    
    # 2. Load Detail Data from the Master Excel file
    cols = ['Fiscal_Year_Code', 'Fiscal_Period', 'Customer_Code', 'Product_Code', 'Family_Code', 'Channel', 'Sales_MU']
    try:
        fya_det = load_sheet_safely(excel_file, 'FYA_Actual_Detail', 'Fiscal_Year_Code', usecols=cols)
        fyb_det = load_sheet_safely(excel_file, 'FYB_Actual_Detail', 'Fiscal_Year_Code', usecols=cols)
        fyc_det = load_sheet_safely(excel_file, 'FYC_Actual_YTD_Detail', 'Fiscal_Year_Code', usecols=cols)
    except FileNotFoundError:
        st.error(f"Error: Could not find '{excel_file}'. Please ensure it is in the same directory.")
        st.stop()

    # 3. Filter YTD (P1-P2) for correlations
    fya_ytd = fya_det[fya_det['Fiscal_Period'].isin([1, 2])]
    fyb_ytd = fyb_det[fyb_det['Fiscal_Period'].isin([1, 2])]
    fyc_ytd = fyc_det[fyc_det['Fiscal_Period'].isin([1, 2])]

    def get_correlation(df_a, df_b, df_c, group_col):
        a = df_a.groupby(group_col)['Sales_MU'].sum().reset_index().rename(columns={'Sales_MU': 'FYA'})
        b = df_b.groupby(group_col)['Sales_MU'].sum().reset_index().rename(columns={'Sales_MU': 'FYB'})
        c = df_c.groupby(group_col)['Sales_MU'].sum().reset_index().rename(columns={'Sales_MU': 'FYC'})
        merged = a.merge(b, on=group_col, how='outer').merge(c, on=group_col, how='outer').fillna(0)
        return {
            'FYA vs FYB': merged['FYA'].corr(merged['FYB']),
            'FYB vs FYC': merged['FYB'].corr(merged['FYC']),
            'FYA vs FYC': merged['FYA'].corr(merged['FYC'])
        }

    corr_customer = get_correlation(fya_ytd, fyb_ytd, fyc_ytd, 'Customer_Code')
    corr_product = get_correlation(fya_ytd, fyb_ytd, fyc_ytd, 'Product_Code')
    corr_family = get_correlation(fya_ytd, fyb_ytd, fyc_ytd, 'Family_Code')

    df_correlations = pd.DataFrame([
        {'Dimension': 'Customer Mix', **corr_customer},
        {'Dimension': 'Family Mix', **corr_family},
        {'Dimension': 'Product Mix', **corr_product}
    ]).set_index('Dimension')
    
    return df_monthly_excel, df_monthly_csv, df_correlations, fya_det

with st.spinner('Loading data and calculating models...'):
    df_monthly_excel, df_monthly_csv, df_correlations, fya_det = load_data()

# --- 2. LAYOUT: TABS ---
st.title("NorthPeak Foods: Forecasting & Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(["Baseline Validation", "Period 8 Anomaly Analysis", "Linear Regression Forecast"])

# ==========================================
# TAB 1: BASELINE VALIDATION
# ==========================================
with tab1:
    st.markdown("**Objective:** Validate that historical data (FYA & FYB) is structurally and volumentrically consistent with current-year actuals (FYC) to serve as a reliable foundation for predictive modeling.")
    st.info("**Conclusion:** The historical dataset demonstrates exceptional structural stability. The business composition explains >90% of the variance year-over-year at the macro level, confirming the viability of historical run-rate forecasting.")
    st.divider()

    st.header("1. Structural Stability (YTD Mix Correlations)")
    st.dataframe(
        df_correlations.style.format("{:.3f}")
        .background_gradient(cmap='Greens', vmin=0.8, vmax=1.0)
        .set_caption("Pearson Correlation (r) by Dimension (Jan-Feb YTD)"),
        use_container_width=True
    )

    st.header("2. Volumetric Seasonality Overlay (Raw Data)")
    # Using the EXCEL dataframe here so the audience sees the Period 8 anomaly
    fya_plot = df_monthly_excel[df_monthly_excel['Year_Code'] == 'FYA'].copy()
    fyb_plot = df_monthly_excel[df_monthly_excel['Year_Code'] == 'FYB'].copy()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fya_plot['Month_Name'], y=fya_plot['Revenue_kMU'], mode='lines+markers', name='FYA Revenue', line=dict(color='royalblue', width=3)))
    fig1.add_trace(go.Scatter(x=fyb_plot['Month_Name'], y=fyb_plot['Revenue_kMU'], mode='lines+markers', name='FYB Revenue', line=dict(color='darkorange', width=3)))
    
    # Adding a callout annotation to make the drop highly visible
    fig1.add_annotation(x="Aug", y=6058, text="⚠️ P8 Anomaly (See Tab 2)", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="red", font=dict(color="white"))

    fig1.update_layout(xaxis_title="Fiscal Period (Month)", yaxis_title="Revenue (kMU)", hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# TAB 2: PERIOD 8 ANOMALY EXPLORATION
# ==========================================
with tab2:
    st.header("The Period 8 Anomaly: Mix Stability Analysis")
    st.markdown("""
    In the raw database extract for FYA Period 8 (August), total revenue dropped by over 70%. If a genuine market event caused this contraction (e.g., loss of a major client or a severe supply chain disruption), the underlying structural *mix* of the business would have drastically shifted. 
    
    The analysis below demonstrates that the underlying proportions of channel, product, and customer distribution remained nearly identical during the volume contraction. **The mix remained static while the total volume disappeared, suggesting this as an  external shock or an IT/data extraction failure.**
    """)

    view_choice = st.radio("Select Business Dimension to Compare:", ["Channel Mix", "Product Family Mix (Top 5)", "Customer Mix (Top 5)"], horizontal=True)
    dim_map = {"Channel Mix": "Channel", "Product Family Mix (Top 5)": "Family_Code", "Customer Mix (Top 5)": "Customer_Code"}
    selected_col = dim_map[view_choice]

    def prep_mix_data(period_num):
        df_period = fya_det[fya_det['Fiscal_Period'] == period_num]
        grouped = df_period.groupby(selected_col)['Sales_MU'].sum().reset_index()
        p7_totals = fya_det[fya_det['Fiscal_Period'] == 7].groupby(selected_col)['Sales_MU'].sum()
        top_5_items = p7_totals.nlargest(5).index.tolist()
        if selected_col != "Channel" and len(grouped) > 5:
            grouped['Category'] = grouped[selected_col].apply(lambda x: x if x in top_5_items else 'Other')
            grouped = grouped.groupby('Category')['Sales_MU'].sum().reset_index()
            grouped = grouped.rename(columns={'Category': selected_col})
        total_sales = grouped['Sales_MU'].sum()
        grouped['Mix_%'] = grouped['Sales_MU'] / total_sales
        return grouped.sort_values('Sales_MU', ascending=False)

    data_p7 = prep_mix_data(7)
    data_p8 = prep_mix_data(8)
    data_p9 = prep_mix_data(9)

    def make_mix_donut(df, title):
        fig = px.pie(df, values='Sales_MU', names=selected_col, title=title, hole=0.4)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        return fig

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(make_mix_donut(data_p7, "Period 7 (July)"), use_container_width=True)
    with c2: st.plotly_chart(make_mix_donut(data_p8, "Period 8 (August)"), use_container_width=True)
    with c3: st.plotly_chart(make_mix_donut(data_p9, "Period 9 (Sept)"), use_container_width=True)

    st.divider()
    st.subheader(f"Mix Shift Breakdown: {view_choice}")
    
    merge_df = data_p7[[selected_col, 'Mix_%']].rename(columns={'Mix_%': 'P7_Mix'})
    merge_df = merge_df.merge(data_p8[[selected_col, 'Mix_%']].rename(columns={'Mix_%': 'P8_Mix'}), on=selected_col, how='outer')
    merge_df = merge_df.merge(data_p9[[selected_col, 'Mix_%']].rename(columns={'Mix_%': 'P9_Mix'}), on=selected_col, how='outer').fillna(0)
    merge_df['Shift (P7 to P8)'] = merge_df['P8_Mix'] - merge_df['P7_Mix']
    merge_df['Shift (P8 to P9)'] = merge_df['P9_Mix'] - merge_df['P8_Mix']
    merge_df = merge_df.sort_values('P7_Mix', ascending=False)

    st.dataframe(
        merge_df.style.format({'P7_Mix': "{:.1%}", 'P8_Mix': "{:.1%}", 'P9_Mix': "{:.1%}", 'Shift (P7 to P8)': "{:+.1%}", 'Shift (P8 to P9)': "{:+.1%}"})
        .background_gradient(subset=['Shift (P7 to P8)', 'Shift (P8 to P9)'], cmap='RdBu', vmin=-0.05, vmax=0.05),
        use_container_width=True, hide_index=True
    )

# ==========================================
# TAB 3: FORECASTING (LINEAR REGRESSION)
# ==========================================
with tab3:
    st.header("Predictive Forecast & Regression Analysis")
    st.markdown("This model predicts future revenue by mathematically isolating the overarching macro-timeline (Trend) from historical monthly fluctuations (Seasonality).")
    
    # --- MODEL PREPARATION (USING CLEANED CSV DATA) ---
    # We use df_monthly_csv here because Period 8 is correctly imputed
    df_lr = df_monthly_csv.copy()
    
    # Failsafe: Ensure Period 8 is handled (if the raw 6058 value snuck in, correct to 13309)
    if df_lr.loc[7, 'Revenue_kMU'] < 10000:
        df_lr.loc[7, 'Revenue_kMU'] = 13309.9828

    df_lr['Trend'] = np.arange(len(df_lr))
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_lr['Month_Name'] = pd.Categorical(df_lr['Month_Name'], categories=months_order, ordered=True)
    
    # Dummies (drop January to act as the Intercept/Baseline)
    month_dummies = pd.get_dummies(df_lr['Month_Name'], drop_first=True, dtype=float)
    
    X = pd.concat([df_lr[['Trend']], month_dummies], axis=1)
    X = sm.add_constant(X) # Statsmodels requires explicit constant
    y = df_lr['Revenue_kMU']
    
    # Fit the Model via OLS
    model = sm.OLS(y, X).fit()
    df_lr['Predicted_Revenue'] = model.predict(X)
    
    # Calculate overarching accuracy metrics
    rmse = np.sqrt(mean_squared_error(y, df_lr['Predicted_Revenue']))
    std_error = np.sqrt(model.mse_resid) # Residual Standard Error
    
    # Display Accuracy Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.0f} kMU", help="The average distance between the model's prediction and the actual revenue.")
    m2.metric("Residual Standard Error (SE)", f"{std_error:,.0f} kMU", help="The standard deviation of the residuals (error).")
    m3.metric("R-Squared", f"{model.rsquared:.3f}", help="The proportion of variance in revenue explained by Trend and Seasonality.")

    # --- GENERATE FUTURE FORECAST & CONFIDENCE INTERVALS ---
    future_trends = np.arange(len(df_lr), len(df_lr) + 6)
    future_months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    
    df_future = pd.DataFrame({'Trend': future_trends, 'Month_Name': future_months})
    df_future['Month_Name'] = pd.Categorical(df_future['Month_Name'], categories=months_order, ordered=True)
    df_future['Year_Code'] = 'FYC'
    
    future_dummies = pd.get_dummies(df_future['Month_Name'], drop_first=True, dtype=float)
    
    # Align columns
    for col in month_dummies.columns:
        if col not in future_dummies.columns:
            future_dummies[col] = 0.0
            
    X_future = pd.concat([df_future[['Trend']], future_dummies[month_dummies.columns]], axis=1)
    X_future.insert(0, 'const', 1.0) # Add intercept to future data
    
    # Get Predictions and 95% Confidence Intervals (Prediction Intervals)
    predictions = model.get_prediction(X_future)
    pred_summary = predictions.summary_frame(alpha=0.05)
    
    df_future['Forecast_kMU'] = pred_summary['mean']
    df_future['Lower_95'] = pred_summary['obs_ci_lower']
    df_future['Upper_95'] = pred_summary['obs_ci_upper']

    # --- MODEL VISUALIZATION ---
    x_labels_hist = df_lr['Year_Code'] + " " + df_lr['Month_Name'].astype(str)
    x_labels_future = df_future['Year_Code'] + " " + df_future['Month_Name'].astype(str)
    
    fig3 = go.Figure()
    
    # Historical Actuals
    fig3.add_trace(go.Scatter(x=x_labels_hist, y=df_lr['Revenue_kMU'], mode='lines+markers', name='Actual Revenue', line=dict(color='black', width=2), marker=dict(size=6)))
    
    # Historical Fit
    fig3.add_trace(go.Scatter(x=x_labels_hist, y=df_lr['Predicted_Revenue'], mode='lines', name='Model Fit', line=dict(dash='dash', color='royalblue', width=2)))
    
    # Future Forecast (Center Line)
    fig3.add_trace(go.Scatter(x=x_labels_future, y=df_future['Forecast_kMU'], mode='lines+markers', name='Projected Forecast', line=dict(color='firebrick', width=3), marker=dict(size=8)))
    
    # Upper Bound 95%
    fig3.add_trace(go.Scatter(x=x_labels_future, y=df_future['Upper_95'], mode='lines', name='Upper 95% Bound', line=dict(dash='dot', color='rgba(214, 39, 40, 0.5)', width=2)))
    
    # Lower Bound 95% (Includes shading up to the Upper Bound)
    fig3.add_trace(go.Scatter(x=x_labels_future, y=df_future['Lower_95'], mode='lines', name='Lower 95% Bound', fill='tonexty', fillcolor='rgba(214, 39, 40, 0.1)', line=dict(dash='dot', color='rgba(214, 39, 40, 0.5)', width=2)))

    # Add a vertical demarcation line
    fig3.add_vline(x=len(df_lr)-1, line_width=1, line_dash="solid", line_color="gray")
    fig3.add_annotation(x=len(df_lr)-0.8, y=22000, text="Forecast Horizon", showarrow=False, textangle=-90, font=dict(color="gray"))

    fig3.update_layout(title='Revenue Trajectory: Historical Fit vs. Predictive Forecast (with 95% CI)', yaxis_title='Revenue (kMU)', hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.divider()

    # --- REGRESSION OUTPUT TABLE & EXPLANATION ---
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Statistical Output Summary")
        
        # Build the SPSS-style DataFrame
        p_values = [f"< 0.001" if p < 0.001 else f"{p:.3f}" for p in model.pvalues]
        
        summary_df = pd.DataFrame({
            'Beta (Coefficient)': model.params,
            'Std. Error': model.bse,
            'P-Value': p_values,
            'Lower 95% CI': model.conf_int()[0],
            'Upper 95% CI': model.conf_int()[1]
        })
        
        # Rename the index for readability
        index_names = ['Intercept (Jan Baseline)', 'Trend (Monthly Slope)'] + [f'Seasonality: {col}' for col in month_dummies.columns]
        summary_df.index = index_names
        
        # Styling function for formatting
        st.dataframe(
            summary_df.style.format({
                'Beta (Coefficient)': "{:,.2f}",
                'Std. Error': "{:,.2f}",
                'Lower 95% CI': "{:,.2f}",
                'Upper 95% CI': "{:,.2f}"
            }),
            use_container_width=True
        )

    with col2:
        st.subheader("Interpreting the Model")
        st.markdown(f"""
        **The Core Components:**
        * **Intercept ({model.params['const']:,.0f} kMU):** The baseline mathematical starting point. Because January was set as the reference variable, this represents the expected revenue of Month 0 (FYA January) excluding any seasonal modifications.
        * **Trend Beta ({model.params['Trend']:,.0f} kMU):** The overarching directional slope. The model indicates the business run-rate naturally decreases by roughly {abs(model.params['Trend']):.0f} kMU per month, holding seasonality constant.
        * **Seasonal Betas:** These coefficients quantify the expected deviation from the January baseline for any given month (e.g., July typically adds +{model.params['Jul']:,.0f} kMU relative to January).
        
        **Statistical Confidence:**
        * **P-Value:** Indicates the probability that the relationship observed is due to random chance. Variables with a P-Value `< 0.05` are considered highly statistically significant drivers of revenue.
        * **95% Confidence Interval (CI):** We are 95% confident that the true population multiplier falls between the Lower and Upper bounds. Narrower ranges indicate higher predictive certainty.
        """)

    st.divider()

    # --- FORECAST DATA TABLE ---
    st.subheader("6-Month Predictive Forecast Schedule (FYC Mar - Aug)")
    st.markdown("Based on the calculated trend deterioration and historical seasonal multipliers, the following revenues and 95% confidence bounds are projected for the active fiscal year:")
    
    display_forecast = df_future[['Year_Code', 'Month_Name', 'Forecast_kMU', 'Lower_95', 'Upper_95']].copy()
    st.dataframe(
        display_forecast.style.format({
            'Forecast_kMU': "{:,.0f} kMU",
            'Lower_95': "{:,.0f} kMU",
            'Upper_95': "{:,.0f} kMU"
        }),
        use_container_width=True, hide_index=True
    )