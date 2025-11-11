"""
Phase 5: Data Drift Monitoring Dashboard
Streamlit app to visualize and detect data drift across train/val/test periods
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NVDA Trading - Drift Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .drift-detected {
        color: #d62728;
        font-weight: bold;
    }
    .no-drift {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    """Load train, validation, and test datasets"""
    try:
        train_df = pd.read_csv('data/NVDA_train.csv', index_col=0, parse_dates=True)
        val_df = pd.read_csv('data/NVDA_val.csv', index_col=0, parse_dates=True)
        test_df = pd.read_csv('data/NVDA_test.csv', index_col=0, parse_dates=True)
        
        # Add period labels
        train_df['period'] = 'Train'
        val_df['period'] = 'Validation'
        test_df['period'] = 'Test'
        
        # Combine for easy analysis
        full_df = pd.concat([train_df, val_df, test_df])
        
        return train_df, val_df, test_df, full_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_data
def get_feature_columns(df):
    """Get list of feature columns"""
    # Get both normalized and original features
    norm_features = [col for col in df.columns if col.endswith('_norm')]
    orig_features = [col.replace('_norm', '') for col in norm_features]
    return norm_features, orig_features

# ============================================================
# DRIFT DETECTION
# ============================================================

def kolmogorov_smirnov_test(data1, data2, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test for distribution comparison
    
    Returns:
        statistic: KS test statistic
        p_value: p-value
        drift_detected: True if p < alpha
    """
    statistic, p_value = stats.ks_2samp(data1, data2)
    drift_detected = p_value < alpha
    return statistic, p_value, drift_detected

def calculate_drift_metrics(train_df, val_df, test_df, features, alpha=0.05):
    """Calculate drift metrics for all features across periods"""
    
    drift_results = []
    
    for feature in features:
        # Train vs Val
        ks_stat_tv, p_val_tv, drift_tv = kolmogorov_smirnov_test(
            train_df[feature].dropna(),
            val_df[feature].dropna(),
            alpha
        )
        
        # Train vs Test
        ks_stat_tt, p_val_tt, drift_tt = kolmogorov_smirnov_test(
            train_df[feature].dropna(),
            test_df[feature].dropna(),
            alpha
        )
        
        # Val vs Test
        ks_stat_vt, p_val_vt, drift_vt = kolmogorov_smirnov_test(
            val_df[feature].dropna(),
            test_df[feature].dropna(),
            alpha
        )
        
        # Overall drift score (average of KS statistics)
        overall_drift_score = (ks_stat_tv + ks_stat_tt + ks_stat_vt) / 3
        
        drift_results.append({
            'feature': feature.replace('_norm', ''),
            'train_vs_val_ks': ks_stat_tv,
            'train_vs_val_p': p_val_tv,
            'train_vs_val_drift': drift_tv,
            'train_vs_test_ks': ks_stat_tt,
            'train_vs_test_p': p_val_tt,
            'train_vs_test_drift': drift_tt,
            'val_vs_test_ks': ks_stat_vt,
            'val_vs_test_p': p_val_vt,
            'val_vs_test_drift': drift_vt,
            'drift_score': overall_drift_score,
            'any_drift': drift_tv or drift_tt or drift_vt
        })
    
    return pd.DataFrame(drift_results)

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_distribution_comparison(train_df, val_df, test_df, feature):
    """Plot distribution comparison across periods"""
    
    fig = go.Figure()
    
    # Train distribution
    fig.add_trace(go.Histogram(
        x=train_df[feature],
        name='Train',
        opacity=0.6,
        nbinsx=30,
        marker_color='#1f77b4'
    ))
    
    # Validation distribution
    fig.add_trace(go.Histogram(
        x=val_df[feature],
        name='Validation',
        opacity=0.6,
        nbinsx=30,
        marker_color='#ff7f0e'
    ))
    
    # Test distribution
    fig.add_trace(go.Histogram(
        x=test_df[feature],
        name='Test',
        opacity=0.6,
        nbinsx=30,
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title=f'Distribution Comparison: {feature.replace("_norm", "")}',
        xaxis_title='Value',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_timeline_view(full_df, feature):
    """Plot feature values over time"""
    
    fig = go.Figure()
    
    # Get period boundaries
    train_df = full_df[full_df['period'] == 'Train']
    val_df = full_df[full_df['period'] == 'Validation']
    test_df = full_df[full_df['period'] == 'Test']
    
    # Plot each period separately with different colors
    fig.add_trace(go.Scatter(
        x=train_df.index,
        y=train_df[feature],
        mode='lines',
        name='Train',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=val_df.index,
        y=val_df[feature],
        mode='lines',
        name='Validation',
        line=dict(color='#ff7f0e', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df[feature],
        mode='lines',
        name='Test',
        line=dict(color='#2ca02c', width=1.5)
    ))
    
    fig.update_layout(
        title=f'Timeline View: {feature.replace("_norm", "")}',
        xaxis_title='Date',
        yaxis_title='Value',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_rolling_statistics(full_df, feature, window=50):
    """Plot rolling mean and standard deviation"""
    
    rolling_mean = full_df[feature].rolling(window=window).mean()
    rolling_std = full_df[feature].rolling(window=window).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Mean', 'Rolling Std Dev'),
        vertical_spacing=0.15
    )
    
    # Get period data
    train_df = full_df[full_df['period'] == 'Train']
    val_df = full_df[full_df['period'] == 'Validation']
    test_df = full_df[full_df['period'] == 'Test']
    
    # Rolling mean - by period
    train_mean = rolling_mean.loc[train_df.index]
    val_mean = rolling_mean.loc[val_df.index]
    test_mean = rolling_mean.loc[test_df.index]
    
    fig.add_trace(
        go.Scatter(x=train_df.index, y=train_mean, name='Train',
                   line=dict(color='#1f77b4'), showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=val_df.index, y=val_mean, name='Validation',
                   line=dict(color='#ff7f0e'), showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=test_df.index, y=test_mean, name='Test',
                   line=dict(color='#2ca02c'), showlegend=True),
        row=1, col=1
    )
    
    # Rolling std - by period
    train_std = rolling_std.loc[train_df.index]
    val_std = rolling_std.loc[val_df.index]
    test_std = rolling_std.loc[test_df.index]
    
    fig.add_trace(
        go.Scatter(x=train_df.index, y=train_std, name='Train',
                   line=dict(color='#1f77b4'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=val_df.index, y=val_std, name='Validation',
                   line=dict(color='#ff7f0e'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=test_df.index, y=test_std, name='Test',
                   line=dict(color='#2ca02c'), showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'Rolling Statistics: {feature.replace("_norm", "")} (window={window})',
        height=600
    )
    
    return fig

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">📊 NVDA Trading Strategy - Data Drift Monitor</div>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        train_df, val_df, test_df, full_df = load_data()
        norm_features, orig_features = get_feature_columns(train_df)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Alpha level for drift detection
    alpha = st.sidebar.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Lower values = stricter drift detection"
    )
    
    # Feature selection
    st.sidebar.markdown("### 📊 Feature Selection")
    feature_display_names = [f.replace('_norm', '') for f in norm_features]
    
    # Calculate drift metrics
    with st.spinner('Calculating drift metrics...'):
        drift_df = calculate_drift_metrics(train_df, val_df, test_df, norm_features, alpha)
    
    # ============================================================
    # OVERVIEW TAB
    # ============================================================
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Feature Analysis", "📊 Drift Statistics", "💡 Insights"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Dataset stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", f"{len(train_df):,}")
            st.caption(f"{train_df.index[0].date()} to {train_df.index[-1].date()}")
        
        with col2:
            st.metric("Validation Samples", f"{len(val_df):,}")
            st.caption(f"{val_df.index[0].date()} to {val_df.index[-1].date()}")
        
        with col3:
            st.metric("Test Samples", f"{len(test_df):,}")
            st.caption(f"{test_df.index[0].date()} to {test_df.index[-1].date()}")
        
        # Drift summary
        st.subheader("🎯 Drift Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_features = len(drift_df)
            st.metric("Total Features", total_features)
        
        with col2:
            drifted_features = drift_df['any_drift'].sum()
            st.metric("Features with Drift", drifted_features)
        
        with col3:
            drift_percentage = (drifted_features / total_features) * 100
            st.metric("Drift Percentage", f"{drift_percentage:.1f}%")
        
        with col4:
            avg_drift_score = drift_df['drift_score'].mean()
            st.metric("Avg Drift Score", f"{avg_drift_score:.3f}")
        
        # Top drifted features
        st.subheader("⚠️ Top 5 Most Drifted Features")
        
        top_drifted = drift_df.nlargest(5, 'drift_score')[['feature', 'drift_score', 'any_drift']]
        
        for idx, row in top_drifted.iterrows():
            drift_status = "🔴 DRIFT DETECTED" if row['any_drift'] else "🟢 NO DRIFT"
            st.markdown(f"**{row['feature']}** - Score: {row['drift_score']:.3f} - {drift_status}")
    
    # ============================================================
    # FEATURE ANALYSIS TAB
    # ============================================================
    
    with tab2:
        st.header("Feature-Level Analysis")
        
        # Feature selector
        selected_feature_name = st.selectbox(
            "Select Feature to Analyze",
            feature_display_names,
            index=0
        )
        
        # Get corresponding normalized feature
        selected_idx = feature_display_names.index(selected_feature_name)
        selected_feature = norm_features[selected_idx]
        
        # Get drift info for selected feature
        feature_drift = drift_df[drift_df['feature'] == selected_feature_name].iloc[0]
        
        # Display drift status
        if feature_drift['any_drift']:
            st.error(f"⚠️ DRIFT DETECTED for {selected_feature_name}")
        else:
            st.success(f"✅ No significant drift detected for {selected_feature_name}")
        
        # Drift metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Train vs Val",
                f"p={feature_drift['train_vs_val_p']:.4f}",
                delta="Drift" if feature_drift['train_vs_val_drift'] else "OK",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Train vs Test",
                f"p={feature_drift['train_vs_test_p']:.4f}",
                delta="Drift" if feature_drift['train_vs_test_drift'] else "OK",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Val vs Test",
                f"p={feature_drift['val_vs_test_p']:.4f}",
                delta="Drift" if feature_drift['val_vs_test_drift'] else "OK",
                delta_color="inverse"
            )
        
        # Visualizations
        st.subheader("Distribution Comparison")
        fig1 = plot_distribution_comparison(train_df, val_df, test_df, selected_feature)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Timeline View")
        fig2 = plot_timeline_view(full_df, selected_feature)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Rolling Statistics")
        window_size = st.slider("Rolling Window Size", 10, 100, 50)
        fig3 = plot_rolling_statistics(full_df, selected_feature, window=window_size)
        st.plotly_chart(fig3, use_container_width=True)
    
    # ============================================================
    # DRIFT STATISTICS TAB
    # ============================================================
    
    with tab3:
        st.header("Comprehensive Drift Statistics")
        
        # Filter options
        show_drift_only = st.checkbox("Show only features with detected drift", value=False)
        
        # Filter data
        display_df = drift_df[drift_df['any_drift']] if show_drift_only else drift_df
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ['drift_score', 'feature', 'train_vs_test_p'],
            index=0
        )
        
        display_df = display_df.sort_values(sort_by, ascending=False if sort_by == 'drift_score' else True)
        
        # Format dataframe for display
        display_df_formatted = display_df.copy()
        display_df_formatted['any_drift'] = display_df_formatted['any_drift'].map({True: '🔴 Yes', False: '🟢 No'})
        
        # Display table
        st.dataframe(
            display_df_formatted[[
                'feature', 'drift_score', 'any_drift',
                'train_vs_val_p', 'train_vs_test_p', 'val_vs_test_p'
            ]].style.format({
                'drift_score': '{:.4f}',
                'train_vs_val_p': '{:.4f}',
                'train_vs_test_p': '{:.4f}',
                'val_vs_test_p': '{:.4f}'
            }),
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Drift Statistics (CSV)",
            data=csv,
            file_name="drift_statistics.csv",
            mime="text/csv"
        )
    
    # ============================================================
    # INSIGHTS TAB
    # ============================================================
    
    with tab4:
        st.header("💡 Key Insights & Interpretation")
        
        # Calculate insights
        total_features = len(drift_df)
        drifted_features = drift_df['any_drift'].sum()
        drift_percentage = (drifted_features / total_features) * 100
        
        most_drifted = drift_df.nlargest(1, 'drift_score').iloc[0]
        least_drifted = drift_df.nsmallest(1, 'drift_score').iloc[0]
        
        st.subheader("📊 Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Dataset Characteristics:**
            - Total features analyzed: {total_features}
            - Features showing drift: {drifted_features} ({drift_percentage:.1f}%)
            - Features stable: {total_features - drifted_features} ({100-drift_percentage:.1f}%)
            - Average drift score: {drift_df['drift_score'].mean():.4f}
            """)
        
        with col2:
            st.markdown(f"""
            **Most/Least Drifted:**
            - Most drifted: **{most_drifted['feature']}** (score: {most_drifted['drift_score']:.4f})
            - Least drifted: **{least_drifted['feature']}** (score: {least_drifted['drift_score']:.4f})
            - Median drift score: {drift_df['drift_score'].median():.4f}
            """)
        
        st.subheader("🔍 What Does This Mean?")
        
        st.markdown("""
        **Understanding Data Drift:**
        
        Data drift occurs when the statistical properties of features change over time. This can happen due to:
        - **Market regime changes** - Bull vs bear markets, high vs low volatility
        - **Structural changes** - Company fundamentals, industry shifts
        - **Macro events** - Economic conditions, policy changes, global events
        
        **Implications for Trading:**
        """)
        
        if drift_percentage > 50:
            st.warning(f"""
            ⚠️ **High Drift Detected** ({drift_percentage:.1f}% of features)
            
            - Model performance may degrade on recent data
            - Consider retraining with more recent data
            - Monitor model performance closely in production
            - May need to adjust feature engineering approach
            """)
        elif drift_percentage > 25:
            st.info(f"""
            ℹ️ **Moderate Drift Detected** ({drift_percentage:.1f}% of features)
            
            - Some features have changed distribution
            - Model should still perform reasonably
            - Recommend periodic retraining
            - Monitor drifted features closely
            """)
        else:
            st.success(f"""
            ✅ **Low Drift Detected** ({drift_percentage:.1f}% of features)
            
            - Feature distributions are relatively stable
            - Model should generalize well to test period
            - Current feature set appears robust
            - Still recommend periodic monitoring
            """)
        
        st.subheader("📈 Recommendations")
        
        st.markdown("""
        **Based on drift analysis:**
        
        1. **For Production Deployment:**
           - Set up monitoring for the most drifted features
           - Establish drift thresholds for retraining triggers
           - Consider adaptive learning approaches
        
        2. **For Model Improvement:**
           - Investigate highly drifted features for market context
           - Consider removing unstable features
           - Add features that capture regime changes
        
        3. **For Backtesting:**
           - Be cautious about out-of-sample performance
           - Consider walk-forward validation
           - Account for changing market conditions
        """)

if __name__ == "__main__":
    main()