"""
DeFi Credit Scoring System for Aave V2 Protocol
A robust machine learning model that assigns credit scores (0-1000) to wallets
based on historical transaction behavior.
Author: Mohammed Faris Sait 
"""


import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class DeFiCreditScorer:
    """
    A comprehensive credit scoring system for DeFi wallets based on Aave V2 transaction data.
    Fixed to handle Wei amounts and various data formats.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.feature_names = []

    def convert_wei_to_ether(self, wei_amount):
        """Convert Wei amount to Ether for easier processing."""
        try:
            if pd.isna(wei_amount) or wei_amount == '' or wei_amount is None:
                return 0.0

            # Convert to string first, then to float
            wei_str = str(wei_amount)

            # Remove any non-numeric characters except decimal point
            wei_str = ''.join(c for c in wei_str if c.isdigit() or c == '.')

            if wei_str == '':
                return 0.0

            wei_float = float(wei_str)
            ether_amount = wei_float / 1e18  # Convert Wei to Ether

            return ether_amount
        except (ValueError, TypeError, OverflowError):
            return 0.0

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess transaction data from JSON file."""
        print("Loading transaction data...")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'transactions' in data:
                    df = pd.DataFrame(data['transactions'])
                elif 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    # Try to find the first list in the dict
                    for key, value in data.items():
                        if isinstance(value, list):
                            df = pd.DataFrame(value)
                            break
                    else:
                        df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported data format")

            print(f"Initial data shape: {df.shape}")
            print(f"Available columns: {list(df.columns)}")


            if 'actionData' in df.columns:
                df['amount'] = df['actionData'].apply(lambda x: x.get('amount') if isinstance(x, dict) else None)
                df['reserve'] = df['actionData'].apply(lambda x: x.get('assetSymbol') if isinstance(x, dict) else None)

            if 'userWallet' in df.columns:
                df = df.rename(columns={'userWallet': 'user'})

            # Check for required columns and suggest mappings
            required_columns = ['user', 'action', 'reserve', 'amount', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Missing columns: {missing_columns}")

                column_mapping = {
                    'user': ['address', 'wallet', 'account', 'from', 'to', 'userAddress'],
                    'action': ['type', 'event', 'method', 'function', 'eventName'],
                    'reserve': ['token', 'asset', 'symbol', 'currency', 'reserveAddress'],
                    'amount': ['value', 'amount_usd', 'amount_wei', 'volume', 'size'],
                    'timestamp': ['time', 'block_time', 'date', 'created_at', 'blockTime']
                }

                rename_dict = {}
                for required_col in missing_columns:
                    for col in df.columns:
                        if col.lower() in [name.lower() for name in column_mapping.get(required_col, [])]:
                            rename_dict[col] = required_col
                            break

                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    print(f"Renamed columns: {rename_dict}")

            # Fill missing required columns with defaults
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found, creating with default values")
                    if col == 'amount':
                        df[col] = 0
                    elif col == 'timestamp':
                        df[col] = datetime.now().timestamp()
                    else:
                        df[col] = 'unknown'

            # Convert timestamp to datetime
            try:
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

                if df['timestamp'].isna().all():
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            except Exception as e:
                print(f"Warning: Could not parse timestamp: {e}")
                df['timestamp'] = pd.to_datetime(datetime.now())

            # Convert amount from Wei to Ether
            print("Converting Wei amounts to Ether...")
            df['amount'] = df['amount'].apply(self.convert_wei_to_ether)

            # Remove invalid rows
            initial_len = len(df)
            df = df.dropna(subset=['user', 'action', 'timestamp'])
            df = df[df['amount'] >= 0]

            print(f"Removed {initial_len - len(df)} invalid rows")
            print(f"Final data shape: {df.shape}")

            if len(df) == 0:
                raise ValueError("No valid transactions found after processing")

            print(f"Loaded {len(df)} transactions for {df['user'].nunique()} unique wallets")
            print(f"Sample amounts (in Ether): {df['amount'].head().tolist()}")

            return df

        except Exception as e:
            print(f"[❌] Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for credit scoring."""
        print("Engineering features...")

        features = []

        for wallet in df['user'].unique():
            wallet_data = df[df['user'] == wallet].copy()
            wallet_data = wallet_data.sort_values('timestamp')

            feature_dict = {'wallet': wallet}

            # === BASIC TRANSACTION METRICS ===
            feature_dict['total_transactions'] = len(wallet_data)
            feature_dict['unique_actions'] = wallet_data['action'].nunique()
            feature_dict['days_active'] = max(
                (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).days + 1, 1)
            feature_dict['avg_daily_transactions'] = feature_dict['total_transactions'] / feature_dict['days_active']

            # === TRANSACTION VOLUME METRICS ===
            feature_dict['total_volume'] = wallet_data['amount'].sum()
            feature_dict['avg_transaction_size'] = wallet_data['amount'].mean()
            feature_dict['median_transaction_size'] = wallet_data['amount'].median()
            feature_dict['volume_std'] = wallet_data['amount'].std()
            feature_dict['volume_coefficient_variation'] = feature_dict['volume_std'] / max(
                feature_dict['avg_transaction_size'], 1e-10)

            # === ACTION DISTRIBUTION ===
            action_counts = wallet_data['action'].value_counts()
            total_actions = len(wallet_data)

            feature_dict['deposit_ratio'] = action_counts.get('deposit', 0) / total_actions
            feature_dict['borrow_ratio'] = action_counts.get('borrow', 0) / total_actions
            feature_dict['repay_ratio'] = action_counts.get('repay', 0) / total_actions
            feature_dict['redeem_ratio'] = action_counts.get('redeemunderlying', 0) / total_actions
            feature_dict['liquidation_ratio'] = action_counts.get('liquidationcall', 0) / total_actions

            # === RISK MANAGEMENT METRICS ===
            deposits = wallet_data[wallet_data['action'] == 'deposit']['amount'].sum()
            borrows = wallet_data[wallet_data['action'] == 'borrow']['amount'].sum()
            repays = wallet_data[wallet_data['action'] == 'repay']['amount'].sum()

            feature_dict['deposit_to_borrow_ratio'] = deposits / max(borrows, 1e-10)
            feature_dict['repay_to_borrow_ratio'] = repays / max(borrows, 1e-10)
            feature_dict['net_position'] = deposits - borrows + repays

            # === PORTFOLIO DIVERSIFICATION ===
            feature_dict['unique_reserves'] = wallet_data['reserve'].nunique()
            feature_dict['reserve_concentration'] = wallet_data['reserve'].value_counts().iloc[0] / len(wallet_data)

            # === TEMPORAL BEHAVIOR ===
            wallet_data['hour'] = wallet_data['timestamp'].dt.hour
            wallet_data['day_of_week'] = wallet_data['timestamp'].dt.dayofweek

            feature_dict['unique_hours'] = wallet_data['hour'].nunique()
            feature_dict['unique_days_of_week'] = wallet_data['day_of_week'].nunique()
            feature_dict['weekend_ratio'] = sum(wallet_data['day_of_week'] >= 5) / len(wallet_data)

            # Transaction intervals
            if len(wallet_data) > 1:
                intervals = wallet_data['timestamp'].diff().dt.total_seconds().dropna()
                feature_dict['avg_interval_hours'] = intervals.mean() / 3600
                feature_dict['median_interval_hours'] = intervals.median() / 3600
                feature_dict['interval_std'] = intervals.std() / 3600
                feature_dict['very_short_intervals'] = sum(intervals < 300) / len(intervals)  # < 5 minutes
            else:
                feature_dict['avg_interval_hours'] = 0
                feature_dict['median_interval_hours'] = 0
                feature_dict['interval_std'] = 0
                feature_dict['very_short_intervals'] = 0

            # === BEHAVIORAL PATTERNS ===
            actions = wallet_data['action'].tolist()
            if len(actions) > 1:
                consecutive_same = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
                feature_dict['consecutive_same_actions'] = consecutive_same / max(len(actions) - 1, 1)

                # Healthy patterns (deposit -> borrow -> repay)
                healthy_patterns = 0
                for i in range(len(actions) - 2):
                    if actions[i] == 'deposit' and actions[i + 1] == 'borrow' and actions[i + 2] == 'repay':
                        healthy_patterns += 1
                feature_dict['healthy_patterns'] = healthy_patterns / max(len(actions) - 2, 1)
            else:
                feature_dict['consecutive_same_actions'] = 0
                feature_dict['healthy_patterns'] = 0

            # === LIQUIDATION RISK ===
            liquidation_count = action_counts.get('liquidationcall', 0)
            feature_dict['liquidation_count'] = liquidation_count
            feature_dict['liquidation_frequency'] = liquidation_count / feature_dict['days_active']

            # === CONSISTENCY METRICS ===
            if feature_dict['days_active'] > 7:
                wallet_data['date'] = wallet_data['timestamp'].dt.date
                daily_counts = wallet_data.groupby('date').size()
                feature_dict['activity_consistency'] = 1 - (daily_counts.std() / max(daily_counts.mean(), 1))
            else:
                feature_dict['activity_consistency'] = 0

            # Volume consistency
            if len(wallet_data) > 5:
                volume_rolling_std = wallet_data['amount'].rolling(window=5).std().mean()
                feature_dict['volume_consistency'] = 1 / (1 + volume_rolling_std)
            else:
                feature_dict['volume_consistency'] = 0.5

            features.append(feature_dict)

        features_df = pd.DataFrame(features)
        features_df = features_df.fillna(0)

        # Replace any infinite values with 0
        features_df = features_df.replace([np.inf, -np.inf], 0)

        print(f"Engineered {len(features_df.columns) - 1} features for {len(features_df)} wallets")
        return features_df

    def calculate_base_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate base credit scores using rule-based approach."""
        print("Calculating base credit scores...")

        df = features_df.copy()

        # Initialize score at 500 (neutral)
        df['base_score'] = 500

        # === POSITIVE SCORING FACTORS ===

        # Long-term activity (0-100 points)
        df['base_score'] += np.clip(df['days_active'] / 30 * 50, 0, 100)

        # Consistent activity (0-80 points)
        df['base_score'] += df['activity_consistency'] * 80

        # Healthy risk management (0-100 points)
        df['base_score'] += np.clip(df['deposit_to_borrow_ratio'] * 20, 0, 60)
        df['base_score'] += np.clip(df['repay_to_borrow_ratio'] * 40, 0, 40)

        # Portfolio diversification (0-60 points)
        df['base_score'] += np.clip(df['unique_reserves'] * 10, 0, 40)
        df['base_score'] += (1 - df['reserve_concentration']) * 20

        # Balanced action distribution (0-80 points)
        action_balance = 1 - np.abs(df['deposit_ratio'] - df['borrow_ratio'])
        df['base_score'] += action_balance * 40
        df['base_score'] += df['healthy_patterns'] * 40

        # === NEGATIVE SCORING FACTORS ===

        # Bot-like behavior penalties (-200 points)
        df['base_score'] -= df['consecutive_same_actions'] * 100
        df['base_score'] -= df['very_short_intervals'] * 100

        # Liquidation penalties (-300 points)
        df['base_score'] -= df['liquidation_count'] * 50
        df['base_score'] -= df['liquidation_frequency'] * 100

        # Excessive risk penalties (-150 points)
        high_borrow_penalty = np.clip((df['borrow_ratio'] - 0.5) * 200, 0, 100)
        df['base_score'] -= high_borrow_penalty

        low_repay_penalty = np.clip((0.3 - df['repay_ratio']) * 100, 0, 50)
        df['base_score'] -= low_repay_penalty

        # Inactivity penalty (-100 points)
        inactivity_penalty = np.clip((7 - df['days_active']) * 10, 0, 50)
        df['base_score'] -= inactivity_penalty

        # Ensure scores are in valid range
        df['base_score'] = np.clip(df['base_score'], 0, 1000)

        return df

    def detect_anomalies(self, features_df: pd.DataFrame) -> np.ndarray:
        """Detect anomalous wallet behavior using Isolation Forest."""
        print("Detecting anomalous behavior...")

        # Select features for anomaly detection
        anomaly_features = [
            'avg_daily_transactions', 'volume_coefficient_variation',
            'consecutive_same_actions', 'very_short_intervals',
            'liquidation_frequency', 'reserve_concentration'
        ]

        # Ensure all features exist
        existing_features = [f for f in anomaly_features if f in features_df.columns]
        if not existing_features:
            print("Warning: No features available for anomaly detection")
            return np.ones(len(features_df))

        X_anomaly = features_df[existing_features]

        # Handle any remaining infinite or NaN values
        X_anomaly = X_anomaly.replace([np.inf, -np.inf], 0)
        X_anomaly = X_anomaly.fillna(0)

        if X_anomaly.shape[1] == 0:
            return np.ones(len(features_df))

        try:
            X_anomaly_scaled = self.scaler.fit_transform(X_anomaly)
            anomaly_scores = self.isolation_forest.fit_predict(X_anomaly_scaled)
        except Exception as e:
            print(f"Warning: Anomaly detection failed: {e}")
            anomaly_scores = np.ones(len(features_df))

        return anomaly_scores

    def train_ml_model(self, features_df: pd.DataFrame) -> None:
        """Train machine learning model to refine credit scores."""
        print("Training ML refinement model...")

        # Prepare features for ML model
        feature_cols = [col for col in features_df.columns if col not in ['wallet', 'base_score']]
        self.feature_names = feature_cols

        if not feature_cols:
            print("Warning: No features available for ML training")
            return

        X = features_df[feature_cols]
        y = features_df['base_score']

        # Clean data
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

            self.model.fit(X_scaled, y)

            # Print feature importance
            if len(self.feature_names) > 0:
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10))

        except Exception as e:
            print(f"Warning: ML model training failed: {e}")
            self.model = None

    def calculate_final_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final credit scores combining all approaches."""
        print("Calculating final credit scores...")

        df = features_df.copy()

        # Get base scores
        df = self.calculate_base_scores(df)

        # Detect anomalies
        anomaly_scores = self.detect_anomalies(df)

        # Apply anomaly penalty
        anomaly_penalty = (anomaly_scores == -1) * 200  # 200 point penalty for anomalies
        df['base_score'] -= anomaly_penalty

        # Train and apply ML refinement
        self.train_ml_model(df)

        if self.model is not None:
            feature_cols = [col for col in df.columns if col not in ['wallet', 'base_score']]
            if feature_cols:
                X = df[feature_cols]
                X = X.replace([np.inf, -np.inf], 0)
                X = X.fillna(0)

                try:
                    X_scaled = self.scaler.transform(X)
                    ml_scores = self.model.predict(X_scaled)

                    # Combine base score with ML refinement (70% base, 30% ML)
                    df['final_score'] = 0.7 * df['base_score'] + 0.3 * ml_scores
                except Exception as e:
                    print(f"Warning: ML scoring failed: {e}")
                    df['final_score'] = df['base_score']
            else:
                df['final_score'] = df['base_score']
        else:
            df['final_score'] = df['base_score']

        # Final clipping and rounding
        df['final_score'] = np.clip(df['final_score'], 0, 1000).round().astype(int)

        return df

    def generate_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis of wallet scores."""
        print("Generating analysis...")

        analysis = {}

        # Score distribution
        scores = results_df['final_score']
        analysis['score_stats'] = {
            'mean': float(scores.mean()),
            'median': float(scores.median()),
            'std': float(scores.std()),
            'min': int(scores.min()),
            'max': int(scores.max())
        }

        # Score ranges
        ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
                  (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]

        analysis['score_distribution'] = {}
        for start, end in ranges:
            count = len(results_df[(results_df['final_score'] >= start) &
                                   (results_df['final_score'] < end if end < 1000 else results_df[
                                                                                           'final_score'] <= end)])
            analysis['score_distribution'][f'{start}-{end}'] = count

        # Low score analysis (0-300)
        low_score_wallets = results_df[results_df['final_score'] < 300]
        if len(low_score_wallets) > 0:
            analysis['low_score_behavior'] = {
                'count': len(low_score_wallets),
                'avg_liquidations': float(low_score_wallets['liquidation_count'].mean()),
                'avg_consecutive_actions': float(low_score_wallets['consecutive_same_actions'].mean()),
                'avg_very_short_intervals': float(low_score_wallets['very_short_intervals'].mean()),
                'avg_days_active': float(low_score_wallets['days_active'].mean())
            }

        # High score analysis (700-1000)
        high_score_wallets = results_df[results_df['final_score'] >= 700]
        if len(high_score_wallets) > 0:
            analysis['high_score_behavior'] = {
                'count': len(high_score_wallets),
                'avg_days_active': float(high_score_wallets['days_active'].mean()),
                'avg_deposit_to_borrow_ratio': float(high_score_wallets['deposit_to_borrow_ratio'].mean()),
                'avg_repay_to_borrow_ratio': float(high_score_wallets['repay_to_borrow_ratio'].mean()),
                'avg_unique_reserves': float(high_score_wallets['unique_reserves'].mean()),
                'avg_healthy_patterns': float(high_score_wallets['healthy_patterns'].mean())
            }

        return analysis

    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict[str, Any]):
        """Create visualization plots for analysis."""
        print("Creating visualizations...")

        try:
            plt.style.use('default')  # Use default style instead of seaborn
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Score distribution histogram
            axes[0, 0].hist(results_df['final_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Credit Score Distribution')
            axes[0, 0].set_xlabel('Credit Score')
            axes[0, 0].set_ylabel('Number of Wallets')
            axes[0, 0].axvline(analysis['score_stats']['mean'], color='red', linestyle='--',
                               label=f"Mean: {analysis['score_stats']['mean']:.1f}")
            axes[0, 0].legend()

            # Score ranges bar chart
            ranges = list(analysis['score_distribution'].keys())
            counts = list(analysis['score_distribution'].values())
            axes[0, 1].bar(ranges, counts, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Score Distribution by Ranges')
            axes[0, 1].set_xlabel('Score Range')
            axes[0, 1].set_ylabel('Number of Wallets')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Days active vs Credit Score
            axes[1, 0].scatter(results_df['days_active'], results_df['final_score'], alpha=0.6, color='green')
            axes[1, 0].set_title('Days Active vs Credit Score')
            axes[1, 0].set_xlabel('Days Active')
            axes[1, 0].set_ylabel('Credit Score')

            # Liquidation impact
            axes[1, 1].scatter(results_df['liquidation_count'], results_df['final_score'], alpha=0.6, color='orange')
            axes[1, 1].set_title('Liquidation Count vs Credit Score')
            axes[1, 1].set_xlabel('Liquidation Count')
            axes[1, 1].set_ylabel('Credit Score')

            plt.tight_layout()
            plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as credit_score_analysis.png")

        except Exception as e:
            print(f"Warning: Visualization creation failed: {e}")

    def score_wallets(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main function to score wallets from transaction data."""
        print("=== DeFi Credit Scoring System ===\n")

        try:
            # Load and process data
            df = self.load_data(file_path)

            # Engineer features
            features_df = self.engineer_features(df)

            # Calculate final scores
            results_df = self.calculate_final_scores(features_df)

            # Generate analysis
            analysis = self.generate_analysis(results_df)

            # Create visualizations
            self.create_visualizations(results_df, analysis)

            # Prepare output
            output_df = results_df[['wallet', 'final_score']].copy()
            output_df = output_df.sort_values('final_score', ascending=False)

            print(f"\n=== SCORING COMPLETE ===")
            print(f"Processed {len(output_df)} wallets")
            print(f"Average Credit Score: {analysis['score_stats']['mean']:.1f}")
            print(f"Score Range: {analysis['score_stats']['min']} - {analysis['score_stats']['max']}")

            return output_df, analysis

        except Exception as e:
            print(f"Error in scoring process: {e}")
            raise


def main():
    """Main execution function."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python fixed_credit_scorer.py <path_to_transactions.json>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        # Initialize scorer
        scorer = DeFiCreditScorer()

        # Score wallets
        results_df, analysis = scorer.score_wallets(file_path)

        # Save results
        results_df.to_csv('wallet_credit_scores.csv', index=False)
        print("Results saved to wallet_credit_scores.csv")

        # Save analysis
        with open('analysis_results.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print("Analysis saved to analysis_results.json")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
