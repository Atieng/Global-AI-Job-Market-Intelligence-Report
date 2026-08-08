"""
=============================================================
AI JOBS DATASET — VISUALIZATION MODULE
=============================================================
 
Sections
--------
1. Univariate Analysis   -> plot_univariate_dashboard(df)
2. Bivariate Analysis    -> plot_bivariate_dashboard(df)
3. Multivariate Analysis -> plot_correlation_dashboard(df)
                             plot_pairplot_dashboard(df)
4. Time Series Analysis  -> plot_timeseries_dashboard(df)
5. Skill Analysis        -> plot_skill_dashboard(df)

Usage
-----
    from src.visualization import *

    run_all_visualizations(df)

Or call each dashboard individually:

    plot_univariate_dashboard(df)
    plot_bivariate_dashboard(df)
=============================================================
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================
# STYLING
# =============================================================
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})

COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff',
          '#ffa657', '#79c0ff', '#56d364', '#ff7b72']

PRIMARY_COLOR = '#58a6ff'
SECONDARY_COLOR = '#f78166'
SKILL_COLOR = '#3fb950'

OUTPUT_DIR = 'figures'

 

def _new_dashboard(nrows, ncols, figsize, suptitle):
    """
    Create a new multi-panel dashboard figure with a styled title.

    Returns
    -------
    fig : matplotlib Figure
    axes : flat list of Axes (length nrows * ncols)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold',
                 color=PRIMARY_COLOR, y=1.02)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    return fig, axes


def _style_panel(ax, title, xlabel, ylabel, rotate_x=0):
    """Apply consistent per-panel styling: title, labels, rotation."""
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_x:
        ax.tick_params(axis='x', rotation=rotate_x)


def _hide_unused(axes, used_count):
    """Hide any leftover empty panels in a grid that aren't used."""
    for ax in axes[used_count:]:
        ax.axis('off')


def _save_dashboard(fig, filename, save):
    """Save a dashboard figure to OUTPUT_DIR/filename.png if save=True."""
    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"Saved dashboard: {path}")
    plt.show()


def _annotate_bars(ax, bars, values, fmt='{:.0f}'):
    """Add value labels above bars."""
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt.format(val), ha='center', va='bottom', fontsize=8)


# =============================================================
# 1. UNIVARIATE ANALYSIS DASHBOARD
 

def plot_univariate_dashboard(df, save=True, filename='fig1_univariate_overview.png'):
    """
    Plot a 4x3 dashboard of univariate distributions:
    salary distribution, salary boxplot, experience level count,
    employment type count, company size count, remote ratio,
    industry distribution, education level, company location
    (top 10), and skill count distribution.
    """
    fig, axes = _new_dashboard(4, 3, (18, 20), 'UNIVARIATE ANALYSIS — AI JOBS DATASET')

    # 1. Salary distribution
    ax = axes[0]
    sns.histplot(df['salary_usd'], kde=True, color=PRIMARY_COLOR, bins=40, ax=ax)
    _style_panel(ax, 'Salary Distribution', 'Salary (USD)', 'Frequency')

    # 2. Salary boxplot
    ax = axes[1]
    sns.boxplot(x=df['salary_usd'], color=PRIMARY_COLOR, ax=ax)
    _style_panel(ax, 'Salary Boxplot', 'Salary (USD)', '')

    # 3. Experience level count
    ax = axes[2]
    order = df['experience_level'].value_counts().index
    sns.countplot(x='experience_level', data=df, order=order,
                  color=COLORS[2], ax=ax)
    _style_panel(ax, 'Experience Level Count', 'Experience Level', 'Count', rotate_x=20)

    # 4. Employment type count
    ax = axes[3]
    order = df['employment_type'].value_counts().index
    sns.countplot(x='employment_type', data=df, order=order,
                  color=COLORS[3], ax=ax)
    _style_panel(ax, 'Employment Type Count', 'Employment Type', 'Count', rotate_x=20)

    # 5. Company size count
    ax = axes[4]
    order = df['company_size'].value_counts().index
    sns.countplot(x='company_size', data=df, order=order,
                  color=COLORS[4], ax=ax)
    _style_panel(ax, 'Company Size Count', 'Company Size', 'Count')

    # 6. Remote ratio distribution
    ax = axes[5]
    order = sorted(df['remote_ratio'].dropna().unique())
    sns.countplot(x='remote_ratio', data=df, order=order,
                  color=COLORS[5], ax=ax)
    _style_panel(ax, 'Remote Ratio Distribution', 'Remote Ratio (%)', 'Count')

    # 7. Industry distribution
    ax = axes[6]
    order = df['industry'].value_counts().index
    sns.countplot(y='industry', data=df, order=order, color=COLORS[6], ax=ax)
    _style_panel(ax, 'Industry Distribution', 'Count', 'Industry')

    # 8. Education level distribution
    ax = axes[7]
    order = df['education_required'].value_counts().index
    sns.countplot(x='education_required', data=df, order=order,
                  color=COLORS[7], ax=ax)
    _style_panel(ax, 'Education Level Distribution', 'Education Level', 'Count', rotate_x=20)

    # 9. Company location distribution (top 10)
    ax = axes[8]
    top_locations = df['company_location'].value_counts().nlargest(10).index
    subset = df[df['company_location'].isin(top_locations)]
    sns.countplot(y='company_location', data=subset, order=top_locations,
                  color=COLORS[0], ax=ax)
    _style_panel(ax, 'Top 10 Company Locations', 'Count', 'Location')

    # 10. Skill count distribution
    ax = axes[9]
    sns.histplot(df['skill_count'], discrete=True, color=SKILL_COLOR, ax=ax)
    _style_panel(ax, 'Skill Count Distribution', 'Number of Required Skills', 'Frequency')

    _hide_unused(axes, 10)
    _save_dashboard(fig, filename, save)
    return fig


# =============================================================
# 2. BIVARIATE ANALYSIS DASHBOARD
 

def plot_bivariate_dashboard(df, save=True, filename='fig2_bivariate_overview.png'):
    """
    Plot a 4x3 dashboard of bivariate relationships against salary:
    experience (scatter), experience level, employment type,
    company size, top-10 industries, remote ratio, education level,
    benefits score, job description length, and skill count.
    """
    fig, axes = _new_dashboard(4, 3, (18, 20), 'BIVARIATE ANALYSIS — SALARY RELATIONSHIPS')

    # 1. Salary vs Experience
    ax = axes[0]
    sns.scatterplot(x='years_experience', y='salary_usd', data=df,
                     color=PRIMARY_COLOR, alpha=0.5, ax=ax)
    _style_panel(ax, 'Salary vs. Years of Experience', 'Years of Experience', 'Salary (USD)')

    # 2. Salary by Experience Level
    ax = axes[1]
    order = df.groupby('experience_level')['salary_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='experience_level', y='salary_usd', data=df, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Experience Level', 'Experience Level', 'Salary (USD)', rotate_x=20)

    # 3. Salary by Employment Type
    ax = axes[2]
    order = df.groupby('employment_type')['salary_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='employment_type', y='salary_usd', data=df, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Employment Type', 'Employment Type', 'Salary (USD)', rotate_x=20)

    # 4. Salary by Company Size
    ax = axes[3]
    order = df.groupby('company_size')['salary_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='company_size', y='salary_usd', data=df, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Company Size', 'Company Size', 'Salary (USD)')

    # 5. Salary by Industry (Top 10)
    ax = axes[4]
    top_industries = df['industry'].value_counts().nlargest(10).index
    subset = df[df['industry'].isin(top_industries)]
    order = subset.groupby('industry')['salary_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='industry', y='salary_usd', data=subset, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Industry (Top 10)', 'Industry', 'Salary (USD)', rotate_x=30)

    # 6. Salary by Remote Ratio
    ax = axes[5]
    order = sorted(df['remote_ratio'].dropna().unique())
    sns.boxplot(x='remote_ratio', y='salary_usd', data=df, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Remote Ratio', 'Remote Ratio (%)', 'Salary (USD)')

    # 7. Salary by Education Level
    ax = axes[6]
    order = df.groupby('education_required')['salary_usd'].median().sort_values(ascending=False).index
    sns.boxplot(x='education_required', y='salary_usd', data=df, order=order,
                palette=COLORS, ax=ax)
    _style_panel(ax, 'Salary by Education Level', 'Education Level', 'Salary (USD)', rotate_x=20)

    # 8. Benefits Score vs Salary
    ax = axes[7]
    sns.scatterplot(x='benefits_score', y='salary_usd', data=df,
                     color=SECONDARY_COLOR, alpha=0.5, ax=ax)
    _style_panel(ax, 'Benefits Score vs. Salary', 'Benefits Score', 'Salary (USD)')

    # 9. Job Description Length vs Salary
    ax = axes[8]
    sns.scatterplot(x='job_description_length', y='salary_usd', data=df,
                     color=PRIMARY_COLOR, alpha=0.5, ax=ax)
    _style_panel(ax, 'Job Description Length vs. Salary', 'Description Length (chars)', 'Salary (USD)')

    # 10. Skill Count vs Salary
    ax = axes[9]
    sns.boxplot(x='skill_count', y='salary_usd', data=df, color=SKILL_COLOR, ax=ax)
    _style_panel(ax, 'Skill Count vs. Salary', 'Number of Required Skills', 'Salary (USD)')

    _hide_unused(axes, 10)
    _save_dashboard(fig, filename, save)
    return fig


# =============================================================
# 3. MULTIVARIATE ANALYSIS DASHBOARDS
# =============================================================
 

def plot_correlation_dashboard(df, numeric_cols=None, save=True,
                                filename='fig3_correlation_heatmap.png'):
    """
    Plot a correlation heatmap dashboard of key numerical features.

    Parameters
    ----------
    numeric_cols : list of str, optional
        Columns to include. Defaults to a curated set of relevant
        numerical columns if not provided.
    """
    if numeric_cols is None:
        numeric_cols = [
            'salary_usd', 'years_experience', 'experience_level_num',
            'company_size_num', 'remote_ratio', 'benefits_score',
            'job_description_length', 'seniority_impact',
            'hiring_window_days', 'application_duration', 'skill_count',
        ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('MULTIVARIATE ANALYSIS — CORRELATION HEATMAP',
                 fontsize=16, fontweight='bold', color=PRIMARY_COLOR, y=1.02)
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.3, cbar_kws={'shrink': 0.8}, ax=ax)
    _style_panel(ax, 'Correlation Matrix of Numerical Features', '', '', rotate_x=30)

    _save_dashboard(fig, filename, save)
    return fig


def plot_pairplot_dashboard(df, numeric_cols=None, save=True,
                             filename='fig4_pairplot.png'):
    """
    Plot a pairplot dashboard of selected numerical variables to
    explore joint distributions and relationships.

    Parameters
    ----------
    numeric_cols : list of str, optional
        Columns to include. Defaults to a curated set of relevant
        numerical columns if not provided.
    """
    if numeric_cols is None:
        numeric_cols = ['salary_usd', 'years_experience', 'benefits_score',
                         'job_description_length', 'skill_count']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    grid = sns.pairplot(df[numeric_cols], diag_kind='kde',
                         plot_kws={'alpha': 0.5, 'color': PRIMARY_COLOR},
                         diag_kws={'color': SECONDARY_COLOR})
    grid.fig.suptitle('MULTIVARIATE ANALYSIS — PAIRPLOT OF NUMERICAL FEATURES',
                       y=1.02, fontsize=16, fontweight='bold', color=PRIMARY_COLOR)
    grid.fig.patch.set_facecolor('#0d1117')
    for ax in grid.axes.flatten():
        if ax is not None:
            ax.set_facecolor('#161b22')

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        grid.fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"Saved dashboard: {path}")
    plt.show()
    return grid


# =============================================================
# 4. TIME SERIES ANALYSIS DASHBOARD
# =============================================================
# Monthly job posting volume and average salary trends.
# =============================================================

def _ensure_datetime(df, column):
    """Return a copy of df with `column` coerced to datetime dtype."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors='coerce')
    return df


def plot_timeseries_dashboard(df, save=True, filename='fig5_timeseries_overview.png'):
    """
    Plot a 1x2 dashboard of time series trends: monthly job posting
    counts and monthly average salary, both derived from posting_date.
    """
    data = _ensure_datetime(df, 'posting_date')

    fig, axes = _new_dashboard(1, 2, (16, 6), 'TIME SERIES ANALYSIS — POSTING TRENDS')

    # 1. Monthly job postings
    ax = axes[0]
    monthly_counts = data.set_index('posting_date').resample('ME').size()
    ax.plot(monthly_counts.index, monthly_counts.values,
            color=PRIMARY_COLOR, marker='o', linewidth=2)
    ax.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.15, color=PRIMARY_COLOR)
    _style_panel(ax, 'Monthly Job Postings', 'Month', 'Number of Postings')

    # 2. Monthly average salary
    ax = axes[1]
    monthly_avg_salary = data.set_index('posting_date')['salary_usd'].resample('ME').mean()
    ax.plot(monthly_avg_salary.index, monthly_avg_salary.values,
            color=SECONDARY_COLOR, marker='o', linewidth=2)
    ax.fill_between(monthly_avg_salary.index, monthly_avg_salary.values, alpha=0.15, color=SECONDARY_COLOR)
    _style_panel(ax, 'Monthly Average Salary', 'Month', 'Average Salary (USD)')

    _save_dashboard(fig, filename, save)
    return fig


# =============================================================
# 5. SKILL ANALYSIS DASHBOARD
# =============================================================
# Counts of postings requiring each key skill (boolean flags).
# =============================================================

def plot_skill_dashboard(df, save=True, filename='fig6_skill_overview.png'):
    """
    Plot a 2x4 dashboard of job postings requiring each key skill:
    Python, SQL, AWS, TensorFlow, PyTorch, NLP, and Docker.
    """
    skill_cols = [
        ('has_python', 'Python'),
        ('has_sql', 'SQL'),
        ('has_aws', 'AWS'),
        ('has_tensorflow', 'TensorFlow'),
        ('has_pytorch', 'PyTorch'),
        ('has_nlp', 'NLP'),
        ('has_docker', 'Docker'),
    ]

    fig, axes = _new_dashboard(2, 4, (18, 10), 'SKILL ANALYSIS — REQUIRED SKILL COUNTS')

    for i, (col, label) in enumerate(skill_cols):
        ax = axes[i]
        counts = df[col].value_counts().reindex([0, 1], fill_value=0)
        bars = ax.bar(['Not Required', 'Required'], counts.values,
                       color=[COLORS[i % len(COLORS)], SKILL_COLOR], edgecolor='none')
        _annotate_bars(ax, bars, counts.values, fmt='{:,.0f}')
        _style_panel(ax, f'{label} Jobs', label, 'Count')

    _hide_unused(axes, len(skill_cols))
    _save_dashboard(fig, filename, save)
    return fig


# =============================================================
# 6. ORCHESTRATION
# =============================================================

def run_all_visualizations(df, save=True):
    """
    Run every dashboard function in this module in a logical order:
    univariate -> bivariate -> multivariate -> time series -> skill
    analysis. Each dashboard is displayed and, by default, saved as
    a PNG under the 'figures/' directory.

        from src.visualization import *
        run_all_visualizations(df)
    """
    plot_univariate_dashboard(df, save=save)
    plot_bivariate_dashboard(df, save=save)
    plot_correlation_dashboard(df, save=save)
    plot_pairplot_dashboard(df, save=save)
    plot_timeseries_dashboard(df, save=save)
    plot_skill_dashboard(df, save=save)