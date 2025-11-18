#!/usr/bin/env python3
"""
Sample Data Generator
=====================

Generates synthetic datasets for testing the retail analytics suite.

Usage:
    python data/generate_sample_data.py

This will create:
    - sample_sales.csv: Time series sales data
    - sample_customers.csv: Customer transaction data for RFM
    - sample_transactions.csv: Transaction data for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)


def generate_sales_data(
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    n_products: int = 5
) -> pd.DataFrame:
    """
    Generate synthetic time series sales data.

    Creates daily sales with:
    - Trend (growth)
    - Yearly seasonality
    - Weekly seasonality
    - Noise
    - Holiday effects

    Args:
        start_date: Start date for the series
        end_date: End date for the series
        n_products: Number of product SKUs

    Returns:
        DataFrame with sales data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    records = []

    for product_id in range(1, n_products + 1):
        # Base sales level
        base_sales = np.random.uniform(100, 500)

        # Trend component (slight growth)
        trend = np.linspace(0, base_sales * 0.3, n_days)

        # Yearly seasonality
        yearly = base_sales * 0.2 * np.sin(
            2 * np.pi * np.arange(n_days) / 365 - np.pi/2
        )

        # Weekly seasonality (weekends higher)
        weekly = np.array([
            0.1 if d.weekday() < 5 else 0.3
            for d in dates
        ]) * base_sales

        # Holiday effects (December boost)
        holidays = np.array([
            0.5 if d.month == 12 else 0
            for d in dates
        ]) * base_sales

        # Noise
        noise = np.random.normal(0, base_sales * 0.1, n_days)

        # Combine components
        sales = base_sales + trend + yearly + weekly + holidays + noise
        sales = np.maximum(sales, 0)  # No negative sales

        for i, date in enumerate(dates):
            records.append({
                'date': date,
                'product_id': f'SKU_{product_id:03d}',
                'sales': round(sales[i], 2),
                'quantity': max(1, int(sales[i] / np.random.uniform(20, 50)))
            })

    df = pd.DataFrame(records)

    # Aggregate to daily total
    daily_sales = df.groupby('date').agg({
        'sales': 'sum',
        'quantity': 'sum'
    }).reset_index()

    return daily_sales


def generate_customer_data(
    n_customers: int = 1000,
    n_transactions: int = 10000,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Generate synthetic customer transaction data for RFM analysis.

    Creates customers with varying:
    - Purchase frequency
    - Average order value
    - Recency patterns

    Args:
        n_customers: Number of unique customers
        n_transactions: Total number of transactions
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with customer transactions
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = (end - start).days

    # Customer profiles
    customer_profiles = {}
    for i in range(1, n_customers + 1):
        customer_profiles[i] = {
            'avg_amount': np.random.lognormal(4, 0.8),  # Skewed distribution
            'frequency': np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.5, 0.3]),
            'recency_bias': np.random.uniform(0, 1)  # Prefer recent or old
        }

    records = []
    transaction_id = 1

    for _ in range(n_transactions):
        # Select customer (weighted by frequency)
        customer_id = np.random.randint(1, n_customers + 1)
        profile = customer_profiles[customer_id]

        # Generate transaction date
        if profile['frequency'] == 'high':
            # More recent transactions
            days_ago = int(np.random.exponential(30))
        elif profile['frequency'] == 'medium':
            days_ago = int(np.random.exponential(90))
        else:
            days_ago = int(np.random.exponential(180))

        days_ago = min(days_ago, date_range)
        txn_date = end - timedelta(days=days_ago)

        # Generate amount
        amount = max(1, np.random.normal(profile['avg_amount'], profile['avg_amount'] * 0.3))

        records.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'date': txn_date,
            'amount': round(amount, 2),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Sports']),
            'channel': np.random.choice(['Online', 'Store', 'Mobile'], p=[0.5, 0.3, 0.2])
        })

        transaction_id += 1

    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def generate_transaction_data(
    n_transactions: int = 50000,
    fraud_rate: float = 0.01,
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Generate synthetic transaction data for fraud detection.

    Creates transactions with:
    - Normal transaction patterns
    - Injected fraudulent transactions with anomalous patterns

    Args:
        n_transactions: Total number of transactions
        fraud_rate: Proportion of fraudulent transactions
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with transaction data and fraud labels
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = (end - start).days

    n_fraud = int(n_transactions * fraud_rate)
    n_normal = n_transactions - n_fraud

    records = []

    # Generate normal transactions
    for i in range(n_normal):
        # Random timestamp
        days = np.random.randint(0, date_range)
        hours = np.random.choice(range(24), p=_get_hour_distribution())
        minutes = np.random.randint(0, 60)
        timestamp = start + timedelta(days=days, hours=hours, minutes=minutes)

        # Normal amount (lognormal distribution)
        amount = np.random.lognormal(4, 1)

        # Customer ID (normal customers have multiple transactions)
        customer_id = np.random.randint(1, 5001)

        records.append({
            'transaction_id': i + 1,
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_id': np.random.randint(1, 501),
            'category': np.random.choice(['Retail', 'Food', 'Travel', 'Entertainment', 'Other']),
            'is_fraud': 0
        })

    # Generate fraudulent transactions
    for i in range(n_fraud):
        # Fraud patterns
        pattern = np.random.choice(['high_amount', 'unusual_time', 'rapid', 'new_merchant'])

        days = np.random.randint(0, date_range)
        timestamp = start + timedelta(days=days)

        if pattern == 'high_amount':
            # Unusually high amount
            amount = np.random.uniform(1000, 10000)
            hours = np.random.choice(range(24), p=_get_hour_distribution())
        elif pattern == 'unusual_time':
            # Transaction at unusual hour (2-5 AM)
            amount = np.random.lognormal(4, 1)
            hours = np.random.randint(2, 6)
        elif pattern == 'rapid':
            # Multiple rapid transactions
            amount = np.random.lognormal(4, 1) * 0.5  # Smaller amounts
            hours = np.random.randint(0, 24)
        else:  # new_merchant
            # New/rare merchant
            amount = np.random.lognormal(4.5, 1.2)
            hours = np.random.choice(range(24), p=_get_hour_distribution())

        timestamp = timestamp + timedelta(hours=hours, minutes=np.random.randint(0, 60))

        records.append({
            'transaction_id': n_normal + i + 1,
            'customer_id': np.random.randint(1, 5001),
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_id': np.random.randint(1, 501) if pattern != 'new_merchant' else np.random.randint(5000, 6000),
            'category': np.random.choice(['Retail', 'Food', 'Travel', 'Entertainment', 'Other']),
            'is_fraud': 1
        })

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = range(1, len(df) + 1)

    return df


def _get_hour_distribution():
    """Get realistic hour distribution for transactions."""
    # Peak during business hours, lower at night
    probs = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
        0.03, 0.05, 0.07, 0.08, 0.08, 0.08,  # 6-11
        0.08, 0.07, 0.07, 0.06, 0.06, 0.05,  # 12-17
        0.05, 0.04, 0.03, 0.02, 0.01, 0.01   # 18-23
    ]
    return np.array(probs) / sum(probs)


def main():
    """Generate all sample datasets."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating sample datasets...")

    # Sales data
    print("  - Generating sales data...")
    sales_df = generate_sales_data()
    sales_path = os.path.join(script_dir, 'sample_sales.csv')
    sales_df.to_csv(sales_path, index=False)
    print(f"    Saved {len(sales_df)} records to {sales_path}")

    # Customer data
    print("  - Generating customer data...")
    customers_df = generate_customer_data()
    customers_path = os.path.join(script_dir, 'sample_customers.csv')
    customers_df.to_csv(customers_path, index=False)
    print(f"    Saved {len(customers_df)} records to {customers_path}")

    # Transaction data
    print("  - Generating transaction data...")
    transactions_df = generate_transaction_data()
    transactions_path = os.path.join(script_dir, 'sample_transactions.csv')
    transactions_df.to_csv(transactions_path, index=False)
    print(f"    Saved {len(transactions_df)} records to {transactions_path}")

    # Also generate a smaller test set
    print("  - Generating small test datasets...")

    small_sales = generate_sales_data(
        start_date='2024-01-01',
        end_date='2024-06-30',
        n_products=2
    )
    small_sales.to_csv(os.path.join(script_dir, 'test_sales_small.csv'), index=False)

    small_customers = generate_customer_data(
        n_customers=100,
        n_transactions=500
    )
    small_customers.to_csv(os.path.join(script_dir, 'test_customers_small.csv'), index=False)

    small_transactions = generate_transaction_data(
        n_transactions=1000,
        fraud_rate=0.02
    )
    small_transactions.to_csv(os.path.join(script_dir, 'test_transactions_small.csv'), index=False)

    print("\nSample data generation complete!")
    print("\nDataset Summary:")
    print(f"  Sales: {len(sales_df)} daily records")
    print(f"  Customers: {len(customers_df)} transactions, {customers_df['customer_id'].nunique()} customers")
    print(f"  Transactions: {len(transactions_df)} records, {transactions_df['is_fraud'].sum()} fraudulent")


if __name__ == '__main__':
    main()
