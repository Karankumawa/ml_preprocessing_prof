import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict

def generate_report(results: List[Dict], problem_type: str = 'classification'):
    """
    Generate comprehensive report with visualizations and summary statistics.
    
    Parameters:
    -----------
    results : List[Dict]
        List of result dictionaries from evaluate_preprocessors
    problem_type : str
        Type of problem: 'classification' or 'regression'
    """
    if not results:
        print("No results to report!")
        return
    
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("ML PREPROCESSING PROFILER RESULTS")
    print("="*80)
    
    # Sort by score (descending for both classification and regression)
    df_sorted = df.sort_values('score', ascending=False)
    
    # Display top 10 results
    print(f"\nTop 10 Preprocessing Combinations ({problem_type.title()}):")
    print("-" * 80)
    
    display_cols = ['combination', 'score', 'time']
    if problem_type == 'classification':
        score_col = 'score'
        score_name = 'Accuracy'
    else:
        score_col = 'score'
        score_name = 'R² Score'
    
    top_results = df_sorted[display_cols].head(10)
    top_results.columns = ['Preprocessing Pipeline', score_name, 'Time (s)']
    print(top_results.to_string(index=False, float_format='%.4f'))
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print("-" * 40)
    print(f"Best {score_name}: {df_sorted[score_col].iloc[0]:.4f}")
    print(f"Worst {score_name}: {df_sorted[score_col].iloc[-1]:.4f}")
    print(f"Mean {score_name}: {df_sorted[score_col].mean():.4f}")
    print(f"Std {score_name}: {df_sorted[score_col].std():.4f}")
    
    # Generate visualizations
    create_performance_plots(df, problem_type)
    create_time_analysis_plots(df)
    
    print(f"\nReport generated successfully! Check the plots above for detailed analysis.")

def create_performance_plots(df: pd.DataFrame, problem_type: str):
    """Create performance comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Preprocessing Performance Analysis ({problem_type.title()})', fontsize=16)
    
    # 1. Score distribution
    axes[0, 0].hist(df['score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top 10 combinations bar plot
    top_10 = df.nlargest(10, 'score')
    axes[0, 1].barh(range(len(top_10)), top_10['score'], color='lightgreen')
    axes[0, 1].set_yticks(range(len(top_10)))
    axes[0, 1].set_yticklabels([f"{row['scaler']}+{row['encoder']}+{row['imputer']}" 
                               for _, row in top_10.iterrows()], fontsize=8)
    axes[0, 1].set_title('Top 10 Preprocessing Combinations')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scaler comparison
    scaler_performance = df.groupby('scaler')['score'].agg(['mean', 'std']).reset_index()
    axes[1, 0].bar(scaler_performance['scaler'], scaler_performance['mean'], 
                   yerr=scaler_performance['std'], capsize=5, color='orange', alpha=0.7)
    axes[1, 0].set_title('Average Performance by Scaler')
    axes[1, 0].set_xlabel('Scaler')
    axes[1, 0].set_ylabel('Mean Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Score vs Time scatter plot
    axes[1, 1].scatter(df['time'], df['score'], alpha=0.6, color='purple')
    axes[1, 1].set_title('Score vs Processing Time')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_time_analysis_plots(df: pd.DataFrame):
    """Create time analysis plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Processing Time Analysis', fontsize=14)
    
    # 1. Time distribution
    axes[0].hist(df['time'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0].set_title('Processing Time Distribution')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Average time by preprocessing type
    scaler_time = df.groupby('scaler')['time'].mean()
    encoder_time = df.groupby('encoder')['time'].mean()
    imputer_time = df.groupby('imputer')['time'].mean()
    
    # Combine all preprocessing types
    all_times = pd.concat([
        scaler_time.rename('Scaler'),
        encoder_time.rename('Encoder'), 
        imputer_time.rename('Imputer')
    ])
    
    axes[1].bar(range(len(all_times)), all_times.values, color=['skyblue', 'lightgreen', 'orange'])
    axes[1].set_title('Average Processing Time by Preprocessing Type')
    axes[1].set_xlabel('Preprocessing Type')
    axes[1].set_ylabel('Average Time (seconds)')
    axes[1].set_xticks(range(len(all_times)))
    axes[1].set_xticklabels(all_times.index, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_latex_table(results: List[Dict], problem_type: str = 'classification') -> str:
    """Generate LaTeX table for academic papers."""
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)
    
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Preprocessing Pipeline Performance Comparison}
\label{tab:preprocessing_results}
\begin{tabular}{lccc}
\toprule
Pipeline & Score & Time (s) & Rank \\
\midrule
"""
    
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        pipeline = f"{row['scaler']} + {row['encoder']} + {row['imputer']}"
        score = f"{row['score']:.4f}"
        time = f"{row['time']:.3f}"
        latex_table += f"{pipeline} & {score} & {time} & {i} \\\\\n"
    
    latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex_table
