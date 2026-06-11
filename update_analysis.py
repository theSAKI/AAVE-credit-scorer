"""
Analysis Report Generator
Updates the analysis.md template (with unique placeholders) using a simple
and robust dictionary-based replacement method.
Author: Mohammed Faris Sait 
"""

import json
import pandas as pd
from datetime import datetime

# ... (load_results function remains the same) ...
def load_results():
    """Load scoring results and analysis data from generated files."""
    try:
        scores_df = pd.read_csv('wallet_credit_scores.csv')
        with open('analysis_results.json', 'r') as f:
            analysis = json.load(f)
        return scores_df, analysis
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure credit_scorer.py has been run successfully.")
        return None, None


def update_analysis_file(scores_df, analysis):
    """Update analysis.md using simple, direct placeholder replacement."""
    with open('analysis.md', 'r') as f:
        content = f.read()

    total_wallets = len(scores_df)
    stats = analysis.get('score_stats', {})
    dist = analysis.get('score_distribution', {})
    low_b = analysis.get('low_score_behavior', {})
    high_b = analysis.get('high_score_behavior', {})

    # ADD THE 3 LINES BELOW
    replacements = {
        "[OVERVIEW_TXN_PERIOD]": "Sample Data",
        "[OVERVIEW_TOTAL_TXNS]": "Not Calculated",
        "[OVERVIEW_AVG_TXNS]": "Not Calculated",
        # --- The rest of the dictionary remains the same ---
        "[OVERVIEW_TOTAL_WALLETS]": str(total_wallets),
        "[STATS_MEAN_SCORE]": f"{stats.get('mean', 0):.1f}",
        "[STATS_MEDIAN_SCORE]": f"{stats.get('median', 0):.0f}",
        "[STATS_STD_DEV]": f"{stats.get('std', 0):.1f}",
        "[STATS_SCORE_RANGE]": f"{stats.get('min', 0)} - {stats.get('max', 0)}",
        "[LOW_SCORE_COUNT]": str(low_b.get('count', 'N/A')),
        "[LOW_SCORE_AVG_DAYS]": f"{low_b.get('avg_days_active', 0):.1f}",
        "[LOW_SCORE_AVG_LIQS]": f"{low_b.get('avg_liquidations', 0):.2f}",
        "[LOW_SCORE_AVG_CONSEC_ACTIONS]": f"{low_b.get('avg_consecutive_actions', 0):.1%}",
        "[LOW_SCORE_AVG_SHORT_INTERVALS]": f"{low_b.get('avg_very_short_intervals', 0):.1%}",
        "[HIGH_SCORE_COUNT]": str(high_b.get('count', 'N/A')),
        "[HIGH_SCORE_AVG_DAYS]": f"{high_b.get('avg_days_active', 0):.1f}",
        "[HIGH_SCORE_DEPOSIT_RATIO]": f"{high_b.get('avg_deposit_to_borrow_ratio', 0):,.0f}",
        "[HIGH_SCORE_REPAY_RATIO]": f"{high_b.get('avg_repay_to_borrow_ratio', 0):,.0f}",
        "[HIGH_SCORE_UNIQUE_RESERVES]": f"{high_b.get('avg_unique_reserves', 0):.1f}",
        "[HIGH_SCORE_HEALTHY_PATTERNS]": f"{high_b.get('avg_healthy_patterns', 0):.1%}",
    }

    # Perform all text replacements.
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    # ... (The rest of the script remains the same) ...
    # Handle the distribution table separately.
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '[TBF]' in line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 4:
                score_range = parts[1]
                if score_range in dist:
                    count = dist.get(score_range, 0)
                    percentage = (count / total_wallets) * 100 if total_wallets > 0 else 0
                    new_lines.append(f"| {parts[1]:<11} | {count:<5} | {f'{percentage:.1f}%':<10} | {parts[4]:<14} |")
                else:
                    new_lines.append(line)
            else:
                 new_lines.append(line)
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    # Update timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = content.replace(
        "*This analysis will be automatically updated when the scoring system is run with actual transaction data.*",
        f"*Analysis generated on {timestamp} using actual transaction data.*"
    )

    with open('analysis_updated.md', 'w') as f:
        f.write(content)

    print("✅ Analysis updated successfully!")
    print("📄 Output written to: analysis_updated.md")


def main():
    """Main execution function."""
    scores_df, analysis = load_results()
    if scores_df is not None and analysis is not None:
        update_analysis_file(scores_df, analysis)

if __name__ == "__main__":
    main()

