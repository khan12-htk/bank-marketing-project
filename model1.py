import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = r'bank_data.csv'
MODEL_PATH = 'xgb_model.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_CLUSTERS = 4

# Core features
NUMERIC_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


def load_and_describe(path: str) -> pd.DataFrame:
    """Load data, log shape, head, info, and basic plots."""
    df = pd.read_csv(path, engine='python')
    logger.info(f'Data shape: {df.shape}')
    logger.info('First five rows:')
    logger.info(df.head().to_string())

    # Capture df.info() output
    buf = io.StringIO()
    df.info(buf=buf)
    logger.info(buf.getvalue())

    logger.info('Descriptive statistics:')
    logger.info(df.describe(include='all').to_string())

    if df.isnull().sum().any():
        logger.info('Missing values:')
        logger.info(df.isnull().sum()[lambda x: x > 0].to_string())

    # Plot actual subscription distribution
    df['y_bin'] = df['y'].map({'no': 0, 'yes': 1})
    plt.figure(figsize=(6,4))
    sns.countplot(x='y_bin', data=df)
    plt.title('Term Deposit Subscription (Actual)')
    plt.xlabel('Subscribed (1) vs Not (0)')
    plt.ylabel('Count')
    plt.show()

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: age buckets, balance bins, recent contact flag."""
    df = df.copy()
    # Age buckets
    df['age_bucket'] = pd.cut(df['age'], bins=[0,25,40,60,100], labels=['<=25','26-40','41-60','61+'])
    # Balance bins (quantile)
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df['balance_bin'] = kb.fit_transform(df[['balance']]).astype(int)
    # Recent contact flag
    df['contact_recent'] = (df['pdays'] < 10).astype(int)
    return df


def build_pipeline(numeric_features, categorical_features):
    """Create preprocessing + XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline


def cluster_personas(X: pd.DataFrame) -> np.ndarray:
    """Cluster customers into personas."""
    km = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE)
    return km.fit_predict(X[NUMERIC_FEATURES])


def train_and_deploy(df: pd.DataFrame):
    """Full pipeline: FE, split, train, full+test predictions, metrics, plots."""
    # Feature engineering
    df_fe = feature_engineering(df)
    df_fe['y_bin'] = df_fe['y'].map({'no': 0, 'yes': 1})
    df_fe['persona'] = cluster_personas(df_fe)

    # Prepare features & target
    X = df_fe.drop(columns=['y', 'y_bin'])
    y = df_fe['y_bin']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Build & train pipeline
    cat_feats = [c for c in X.columns if c not in NUMERIC_FEATURES]
    pipeline = build_pipeline(NUMERIC_FEATURES, cat_feats)
    logger.info('Training XGBoost model...')
    pipeline.fit(X_train, y_train)
    logger.info('Training completed.')

    # ——— New: Predict on the *entire* dataset (all ~45k rows) ———
    X_full = df_fe.drop(columns=['y', 'y_bin'])
    y_full_pred = pipeline.predict(X_full)
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_full_pred)
    plt.title('Predicted Subscription Distribution (All Customers)')
    plt.xlabel('Predicted Subscribe (1) vs Not (0)')
    plt.ylabel('Count')
    total_full = len(y_full_pred)
    for p in plt.gca().patches:
        pct = p.get_height() / total_full * 100
        plt.gca().annotate(f'{pct:.1f}%', 
                           (p.get_x() + p.get_width()/2, p.get_height()),
                           ha='center', va='bottom')
    plt.show()
    # ——————————————————————————————————————————————————————————————

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f'Model saved to {MODEL_PATH}')

    # Predictions on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    # Log test metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    logger.info('Test metrics: ' + ', '.join(f"{k}={v:.3f}" for k,v in metrics.items()))

    # Sample raw predictions
    logger.info('Sample predictions (predicted, probability, actual):')
    for i, (pred, proba, actual) in enumerate(zip(y_pred[:10], y_proba[:10], y_test.values[:10])):
        logger.info(f"{i+1:02d}: pred={pred}, proba={proba:.3f}, actual={actual}")

    # Save full test‐set results
    results = X_test.copy()
    results['predicted'] = y_pred
    results['proba'] = y_proba
    results['actual'] = y_test.values
    out_path = os.path.join(os.getcwd(), 'predictions.csv')
    try:
        results.to_csv(out_path, index=False)
        logger.info(f'Saved all test-set predictions to {out_path}')
    except Exception as e:
        logger.error(f'Failed to save predictions.csv: {e}')

    # Plot predicted subscription distribution (test set)
    plt.figure(figsize=(6,4))
    sns.countplot(x='predicted', data=results)
    plt.title('Predicted Subscription Distribution (Test Set)')
    plt.xlabel('Predicted Subscribed (1) vs Not (0)')
    plt.ylabel('Count')
    total = len(results)
    for p in plt.gca().patches:
        pct = p.get_height() / total * 100
        plt.gca().annotate(f'{pct:.1f}%',
                           (p.get_x() + p.get_width()/2, p.get_height()),
                           ha='center', va='bottom')
    plt.show()

    # Predicted rate by feature
    logger.info('Plotting predicted subscription rate by feature')
    df_results = results.copy()
    for feat in NUMERIC_FEATURES + cat_feats:
        plt.figure(figsize=(6,4))
        if feat in NUMERIC_FEATURES:
            df_results['bin_' + feat] = pd.qcut(df_results[feat], q=10, duplicates='drop')
            grp = df_results.groupby('bin_' + feat)['predicted'].mean()
            grp.plot(kind='bar')
            plt.xlabel(feat + ' (binned)')
        else:
            grp = df_results.groupby(feat)['predicted'].mean().sort_values()
            grp.plot(kind='bar')
            plt.xlabel(feat)
        plt.ylabel('Predicted Subscribe Rate')
        plt.title(f'Predicted Subscription Rate by {feat}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Permutation importance
    logger.info('Computing feature importances...')
    imp = permutation_importance(pipeline, X_test, y_test, n_repeats=10,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({'feature': X_test.columns, 'importance': imp.importances_mean})
    top_imp = imp_df.sort_values('importance', ascending=False).head(10)
    plt.figure(figsize=(8,5))
    plt.barh(top_imp['feature'][::-1], top_imp['importance'][::-1])
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Mean Permutation Importance')
    plt.show()

    # Partial dependence on top feature
    top_feat = top_imp.iloc[0]['feature']
    PartialDependenceDisplay.from_estimator(pipeline, X_test, [top_feat], kind='both')
    plt.show()

    # Business insights
    top20 = results.nlargest(int(0.2*len(results)), 'proba')
    logger.info(f"Top 20% leads avg conv rate: {top20['actual'].mean()*100:.2f}%")
    seg = results.groupby('job').agg(conv_rate=('actual','mean'), count=('actual','size')) \
                 .sort_values('conv_rate', ascending=False)
    logger.info('Conversion rate by job:')
    logger.info(seg.to_string())


def main():
    df = load_and_describe(DATA_PATH)
    train_and_deploy(df)


if __name__ == '__main__':
    main()
