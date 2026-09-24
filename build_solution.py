import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('plots', exist_ok=True)

print("="*70)
print("DEMAND FORECASTING PIPELINE - 1,115 HUBS")
print("="*70)

# -------------------------------------------------------------
# Metric Definition
# -------------------------------------------------------------
def calc_rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def calc_wape(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# -------------------------------------------------------------
# STEP 1: Load each file & check integrity
# -------------------------------------------------------------
print("\n--- STEP 1: Loading Datasets ---")
train = pd.read_csv('orders_train.csv', parse_dates=['Date'])
test = pd.read_csv('orders_test.csv', parse_dates=['Date'])
meta = pd.read_csv('hub_metadata.csv')
sample = pd.read_csv('sample_submission.csv')

print(f"Train Shape: {train.shape}, Date Range: {train['Date'].min().date()} to {train['Date'].max().date()}")
print(f"Test Shape:  {test.shape}, Date Range: {test['Date'].min().date()} to {test['Date'].max().date()}")
print(f"Meta Shape:  {meta.shape}")
print(f"Sample Shape:{sample.shape}")

# Verify test Ids match sample submission in exact same order
id_match = (test['Id'].values == sample['Id'].values).all()
print(f"Verification: Test Ids match sample_submission Ids row-by-row: {id_match}")
assert id_match, "Test IDs do not match sample submission!"

# Closed day verification in train
closed_train_orders = train[train['IsOpen'] == 0]['OrderVolume'].sum()
print(f"Verification: Total orders when IsOpen==0 in train: {closed_train_orders}")
assert closed_train_orders == 0, "Train has non-zero orders when IsOpen==0!"

# Test closed count
test_closed_count = (test['IsOpen'] == 0).sum()
print(f"Verification: Closed rows in test (IsOpen==0): {test_closed_count} (out of {len(test)})")
assert test_closed_count == 6548, f"Expected 6,548 closed rows in test, got {test_closed_count}"

# -------------------------------------------------------------
# STEP 2, 3, 4: Exploratory Data Analysis & Plots
# -------------------------------------------------------------
print("\n--- STEP 2, 3, 4: Generating Exploratory Visualizations ---")

# 1. Volume distribution (raw & log scale)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
open_train = train[train['IsOpen'] == 1]
sns.histplot(open_train['OrderVolume'], bins=60, kde=True, ax=axes[0], color='#1f77b4')
axes[0].set_title('Order Volume Distribution (Raw Scale - Open Hubs)')
axes[0].set_xlabel('Order Volume')

sns.histplot(np.log1p(open_train['OrderVolume']), bins=60, kde=True, ax=axes[1], color='#2ca02c')
axes[1].set_title('Order Volume Distribution (Log1p Scale - Open Hubs)')
axes[1].set_xlabel('log(OrderVolume + 1)')
plt.tight_layout()
plt.savefig('plots/1_volume_distribution.png', dpi=150)
plt.close()

# 2. Weekday pattern & open share by weekday
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
weekday_names = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
train['WeekdayName'] = train['Weekday'].map(weekday_names)
test['WeekdayName'] = test['Weekday'].map(weekday_names)

weekday_vol = open_train.groupby('Weekday')['OrderVolume'].mean()
sns.barplot(x=[weekday_names[i] for i in weekday_vol.index], y=weekday_vol.values, ax=axes[0], palette='Blues_d')
axes[0].set_title('Average Daily Order Volume by Weekday (Open Hubs)')
axes[0].set_ylabel('Mean Order Volume')

weekday_open_rate = train.groupby('Weekday')['IsOpen'].mean()
sns.barplot(x=[weekday_names[i] for i in weekday_open_rate.index], y=weekday_open_rate.values, ax=axes[1], palette='Purples_d')
axes[1].set_title('Hub Open Rate by Weekday')
axes[1].set_ylabel('Share of Open Hubs')
plt.tight_layout()
plt.savefig('plots/2_weekday_seasonality.png', dpi=150)
plt.close()

# 3. Promo Impact & School Closure Impact per Hub
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
promo_overall = open_train.groupby('PromoActive')['OrderVolume'].mean()
sns.barplot(x=['No Promo (0)', 'Promo Active (1)'], y=promo_overall.values, ax=axes[0], palette='Oranges_d')
axes[0].set_title(f'Overall Promo Lift: +{(promo_overall[1]/promo_overall[0]-1)*100:.1f}%')
axes[0].set_ylabel('Mean Order Volume')

school_overall = open_train.groupby('SchoolClosureFlag')['OrderVolume'].mean()
sns.barplot(x=['Regular Day (0)', 'School Closure (1)'], y=school_overall.values, ax=axes[1], palette='Greens_d')
axes[1].set_title(f'School Closure Volume Impact: {(school_overall[1]/school_overall[0]-1)*100:+.1f}%')
axes[1].set_ylabel('Mean Order Volume')
plt.tight_layout()
plt.savefig('plots/3_promo_and_school_impact.png', dpi=150)
plt.close()

# 4. Test Dynamics: Closed Hubs, Promo Windows, School Closure Ramp
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
test_daily = test.groupby('Date').agg(
    closed_hubs=('IsOpen', lambda x: (x == 0).sum()),
    promo_active=('PromoActive', lambda x: (x == 1).sum()),
    school_closures=('SchoolClosureFlag', lambda x: (x == 1).sum())
).reset_index()

axes[0].plot(test_daily['Date'], test_daily['closed_hubs'], color='crimson', marker='o', lw=2)
axes[0].set_title('Test Set: Number of Closed Hubs per Day (Sunday Spikes: ~1,083 Hubs)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_daily['Date'], test_daily['promo_active'], color='darkorange', marker='s', lw=2)
axes[1].set_title('Test Set: Active Promotional Windows (Jun 29-Jul 3, Jul 13-17, Jul 27-31)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(test_daily['Date'], test_daily['school_closures'], color='teal', marker='^', lw=2)
axes[2].set_title('Test Set: School Closure Ramp (Expanding from 286 on Jun 29 to 935 on Jul 31)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/4_test_timeline_dynamics.png', dpi=150)
plt.close()

# 5. Metadata relationships
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hub_median_vol = open_train.groupby('HubID')['OrderVolume'].median().reset_index()
meta_merged = meta.merge(hub_median_vol, on='HubID', how='left')

sns.boxplot(x='HubFormat', y='OrderVolume', data=meta_merged, ax=axes[0], palette='Set2')
axes[0].set_title('Median Hub Volume by Hub Format (1 - 4)')
axes[0].set_ylabel('Median Order Volume')

sns.boxplot(x='AssortmentTier', y='OrderVolume', data=meta_merged, ax=axes[1], palette='Set3')
axes[1].set_title('Median Hub Volume by Assortment Tier (1 - 3)')
axes[1].set_ylabel('Median Order Volume')
plt.tight_layout()
plt.savefig('plots/5_metadata_relationships.png', dpi=150)
plt.close()

print("Plots successfully generated and saved to ./plots/")

# -------------------------------------------------------------
# STEP 5: Feature Engineering Pipeline (Strictly Past-Only)
# -------------------------------------------------------------
print("\n--- STEP 5: Building Feature Pipeline ---")

meta_proc = meta.copy()
meta_proc['log_CompetitorDistance'] = np.log1p(meta_proc['CompetitorDistance'].fillna(meta_proc['CompetitorDistance'].median()))
month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def extract_features(df_target, df_history, meta_df, cutoff_date):
    """
    Extract strictly past-only features for df_target using history before cutoff_date.
    No feature uses any data on or after cutoff_date.
    """
    feat = df_target.copy()
    
    # 1. Date / Calendar features
    feat['Year'] = feat['Date'].dt.year
    feat['Month'] = feat['Date'].dt.month
    feat['DayOfMonth'] = feat['Date'].dt.day
    feat['DayOfYear'] = feat['Date'].dt.dayofyear
    feat['ForecastHorizonDay'] = (feat['Date'] - cutoff_date).dt.days + 1
    
    # 2. Slice strictly past history
    hist = df_history[df_history['Date'] < cutoff_date].copy()
    hist['log_vol'] = np.log1p(hist['OrderVolume'])
    open_hist = hist[hist['IsOpen'] == 1]
    
    # 3. Rolling log volume mean and median over 7, 14, 28, 56, 112 days (closed days count as 0)
    stats_dfs = []
    for w in [7, 14, 28, 56, 112]:
        w_start = cutoff_date - pd.Timedelta(days=w)
        w_hist = hist[hist['Date'] >= w_start]
        agg = w_hist.groupby('HubID')['log_vol'].agg(
            [('mean_log_%d' % w, 'mean'),
             ('median_log_%d' % w, 'median')]
        ).reset_index()
        stats_dfs.append(agg)
        
    hub_stats = stats_dfs[0]
    for agg in stats_dfs[1:]:
        hub_stats = hub_stats.merge(agg, on='HubID', how='outer')
        
    # Momentum: last-28-day mean minus last-56-day mean
    hub_stats['momentum_28_56'] = hub_stats['mean_log_28'] - hub_stats['mean_log_56']
    
    # 4. Hub open-day volume level on last 28 days (for baseline Model a)
    last_28_open = open_hist[open_hist['Date'] >= cutoff_date - pd.Timedelta(days=28)]
    hub_28_open_mean = last_28_open.groupby('HubID')['OrderVolume'].mean().rename('hub_28_open_vol_mean').reset_index()
    hub_stats = hub_stats.merge(hub_28_open_mean, on='HubID', how='left')
    
    # 5. Usual weekday level over last 182 days (open days only)
    w182_start = cutoff_date - pd.Timedelta(days=182)
    hist_182_open = open_hist[open_hist['Date'] >= w182_start]
    hub_weekday_log = hist_182_open.groupby(['HubID', 'Weekday'])['log_vol'].mean().reset_index()
    hub_weekday_log.rename(columns={'log_vol': 'hub_weekday_mean_log'}, inplace=True)
    hub_weekday_vol = hist_182_open.groupby(['HubID', 'Weekday'])['OrderVolume'].mean().reset_index()
    hub_weekday_vol.rename(columns={'OrderVolume': 'hub_weekday_mean_vol'}, inplace=True)
    
    # Fallback overall hub level in case a specific weekday was never open in the window
    hub_182_mean_vol = hist_182_open.groupby('HubID')['OrderVolume'].mean().rename('hub_182_mean_vol_fallback').reset_index()
    hub_182_mean_log = hist_182_open.groupby('HubID')['log_vol'].mean().rename('hub_182_mean_log_fallback').reset_index()
    
    # 6. Promo effect over last 90, 180, and 365 days (mean promo - mean normal)
    for pw in [90, 180, 365]:
        pw_start = cutoff_date - pd.Timedelta(days=pw)
        pw_hist = open_hist[open_hist['Date'] >= pw_start]
        p_stats_vol = pw_hist.groupby(['HubID', 'PromoActive'])['OrderVolume'].mean().unstack()
        p_stats_log = pw_hist.groupby(['HubID', 'PromoActive'])['log_vol'].mean().unstack()
        
        eff_vol = (p_stats_vol[1] - p_stats_vol[0]).fillna(0).rename(f'promo_effect_vol_{pw}').reset_index()
        eff_log = (p_stats_log[1] - p_stats_log[0]).fillna(0).rename(f'promo_effect_log_{pw}').reset_index()
        hub_stats = hub_stats.merge(eff_vol, on='HubID', how='left').merge(eff_log, on='HubID', how='left')
        
    # Standard 365 promo effect aliases
    hub_stats['promo_effect_vol'] = hub_stats['promo_effect_vol_365']
    hub_stats['promo_effect_log'] = hub_stats['promo_effect_log_365']
    
    # 7. School closure effect over last 90, 180, and 365 days
    for sw in [90, 180, 365]:
        sw_start = cutoff_date - pd.Timedelta(days=sw)
        sw_hist = open_hist[open_hist['Date'] >= sw_start]
        s_stats_vol = sw_hist.groupby(['HubID', 'SchoolClosureFlag'])['OrderVolume'].mean().unstack()
        s_stats_log = sw_hist.groupby(['HubID', 'SchoolClosureFlag'])['log_vol'].mean().unstack()
        
        seff_vol = (s_stats_vol[1] - s_stats_vol[0]).fillna(0).rename(f'school_effect_vol_{sw}').reset_index()
        seff_log = (s_stats_log[1] - s_stats_log[0]).fillna(0).rename(f'school_effect_log_{sw}').reset_index()
        hub_stats = hub_stats.merge(seff_vol, on='HubID', how='left').merge(seff_log, on='HubID', how='left')
        
    hub_stats['school_effect_vol'] = hub_stats['school_effect_vol_365']
    hub_stats['school_effect_log'] = hub_stats['school_effect_log_365']
    
    # 8. Recency & hub age features
    last_open_date = open_hist.groupby('HubID')['Date'].max().rename('last_open_date').reset_index()
    first_open_date = open_hist.groupby('HubID')['Date'].min().rename('first_open_date').reset_index()
    
    last_28_data = hist[hist['Date'] >= cutoff_date - pd.Timedelta(days=28)]
    share_open_28 = last_28_data.groupby('HubID')['IsOpen'].mean().rename('share_open_28').reset_index()
    promo_days_28 = last_28_data[last_28_data['PromoActive'] == 1].groupby('HubID')['Date'].nunique().rename('promo_days_28').reset_index()
    closure_days_28 = (28 - last_28_data.groupby('HubID')['IsOpen'].sum()).rename('closure_days_28').reset_index()
    
    hub_stats = hub_stats.merge(last_open_date, on='HubID', how='left') \
                         .merge(first_open_date, on='HubID', how='left') \
                         .merge(share_open_28, on='HubID', how='left') \
                         .merge(promo_days_28, on='HubID', how='left') \
                         .merge(closure_days_28, on='HubID', how='left') \
                         .merge(hub_182_mean_vol, on='HubID', how='left') \
                         .merge(hub_182_mean_log, on='HubID', how='left')
                         
    # Merge onto target frame
    feat = feat.merge(hub_stats, on='HubID', how='left')
    feat = feat.merge(hub_weekday_vol, on=['HubID', 'Weekday'], how='left')
    feat = feat.merge(hub_weekday_log, on=['HubID', 'Weekday'], how='left')
    
    # Impute missing weekday combinations with hub fallback
    feat['hub_weekday_mean_vol'] = feat['hub_weekday_mean_vol'].fillna(feat['hub_182_mean_vol_fallback']).fillna(feat['hub_28_open_vol_mean']).fillna(0)
    feat['hub_weekday_mean_log'] = feat['hub_weekday_mean_log'].fillna(feat['hub_182_mean_log_fallback']).fillna(feat['mean_log_28']).fillna(0)
    
    # Recency deltas
    feat['days_since_last_open'] = (feat['Date'] - feat['last_open_date']).dt.days.fillna(999)
    feat['days_since_first_open'] = (feat['Date'] - feat['first_open_date']).dt.days.fillna(999)
    feat.drop(columns=['last_open_date', 'first_open_date'], inplace=True)
    
    # 9. Merge metadata features
    feat = feat.merge(meta_df, on='HubID', how='left')
    
    # Competitor tenure in months
    feat['comp_open_months'] = (feat['Year'] - feat['CompetitorOpenSinceYear']) * 12 + (feat['Month'] - feat['CompetitorOpenSinceMonth'])
    feat['comp_open_months'] = feat['comp_open_months'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
    feat['is_competitor_open'] = (feat['comp_open_months'] > 0).astype(int)
    
    # Loyalty program tenure in weeks
    feat['loyalty_tenure_weeks'] = (feat['Year'] - feat['LoyaltyProgramSinceYear']) * 52 + (feat['Date'].dt.isocalendar().week - feat['LoyaltyProgramSinceWeek'])
    feat['loyalty_tenure_weeks'] = feat['loyalty_tenure_weeks'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
    
    # Active loyalty month
    def check_loyalty_active(row):
        if row['LoyaltyProgram'] == 1 and pd.notnull(row['LoyaltyProgramInterval']):
            intervals = [month_dict.get(m.strip()) for m in str(row['LoyaltyProgramInterval']).split(',')]
            return 1 if row['Month'] in intervals and row['loyalty_tenure_weeks'] > 0 else 0
        return 0
    feat['is_loyalty_active_month'] = feat.apply(check_loyalty_active, axis=1)
    
    return feat

# Feature column list for modeling
feature_cols = [
    'HubID', 'Weekday', 'Month', 'DayOfMonth', 'DayOfYear', 'ForecastHorizonDay',
    'PromoActive', 'SchoolClosureFlag', 'RegionalHoliday',
    'mean_log_7', 'median_log_7', 'mean_log_14', 'median_log_14',
    'mean_log_28', 'median_log_28', 'mean_log_56', 'median_log_56',
    'mean_log_112', 'median_log_112', 'momentum_28_56',
    'hub_weekday_mean_log', 'hub_weekday_mean_vol',
    'promo_effect_log_90', 'promo_effect_vol_90',
    'promo_effect_log_180', 'promo_effect_vol_180',
    'promo_effect_log_365', 'promo_effect_vol_365',
    'school_effect_log_90', 'school_effect_vol_90',
    'school_effect_log_180', 'school_effect_vol_180',
    'school_effect_log_365', 'school_effect_vol_365',
    'days_since_last_open', 'days_since_first_open',
    'share_open_28', 'promo_days_28', 'closure_days_28',
    'HubFormat', 'AssortmentTier', 'log_CompetitorDistance',
    'comp_open_months', 'is_competitor_open',
    'LoyaltyProgram', 'loyalty_tenure_weeks', 'is_loyalty_active_month'
]

# -------------------------------------------------------------
# STEP 6: Validation Set & Multi-Cutoff Training
# -------------------------------------------------------------
print("\n--- STEP 6: Generating Validation Set (2015-05-09 to 2015-06-19) ---")
val_cutoff = pd.Timestamp('2015-05-09')
val_end = pd.Timestamp('2015-06-19')

val_target = train[(train['Date'] >= val_cutoff) & (train['Date'] <= val_end)].copy()
val_feats = extract_features(val_target, train, meta_proc, val_cutoff)
print(f"Validation dataset created: {val_feats.shape[0]} rows (strictly using data before {val_cutoff.date()})")

print("Building historical training blocks across multiple 42-day forecast windows...")
historical_cutoffs = [
    pd.Timestamp('2015-03-28'),
    pd.Timestamp('2015-02-14'),
    pd.Timestamp('2015-01-03'),
    pd.Timestamp('2014-11-22'),
    pd.Timestamp('2014-10-11'),
    pd.Timestamp('2014-08-30'),
    pd.Timestamp('2014-07-19'),
]

train_blocks = []
for c in historical_cutoffs:
    c_target = train[(train['Date'] >= c) & (train['Date'] < c + pd.Timedelta(days=42))].copy()
    if len(c_target) > 0:
        c_feats = extract_features(c_target, train, meta_proc, c)
        train_blocks.append(c_feats)
        print(f"  Cutoff {c.date()}: {len(c_feats)} rows")

train_df = pd.concat(train_blocks, ignore_index=True)
print(f"Total training samples: {len(train_df)} rows")

# Filter training samples on open days with positive order volume
train_df_open = train_df[(train_df['IsOpen'] == 1) & (train_df['OrderVolume'] > 0)].copy()
val_feats_open = val_feats[(val_feats['IsOpen'] == 1) & (val_feats['OrderVolume'] > 0)].copy()

X_train = train_df_open[feature_cols]
y_train_log = np.log1p(train_df_open['OrderVolume'])

X_val_open = val_feats_open[feature_cols]
y_val_open_log = np.log1p(val_feats_open['OrderVolume'])

X_val_all = val_feats[feature_cols]
y_val_true = val_feats['OrderVolume'].values

# -------------------------------------------------------------
# STEP 6.a: Model (a) - Dumb Baseline
# -------------------------------------------------------------
print("\n--- Evaluating Model (a): Dumb Baseline ---")
# Predict hub's last-28-day open mean; 0 when IsOpen == 0
pred_a = val_feats['hub_28_open_vol_mean'].fillna(0).values.copy()
pred_a[val_feats['IsOpen'] == 0] = 0.0
pred_a = np.clip(pred_a, 0, None)

rmsle_a = calc_rmsle(y_val_true, pred_a)
wape_a = calc_wape(y_val_true, pred_a)
print(f"Model (a) - Dumb Baseline:  RMSLE = {rmsle_a:.5f} | Accuracy = {(1 - wape_a)*100:.2f}%")

# -------------------------------------------------------------
# STEP 6.b: Model (b) - Simple Additive Model
# -------------------------------------------------------------
print("\n--- Evaluating Model (b): Simple Additive Model ---")
# Hub x Weekday level + promo effect + school closure effect, 0 on closed
pred_b = (val_feats['hub_weekday_mean_vol'] + 
          val_feats['PromoActive'] * val_feats['promo_effect_vol_365'] + 
          val_feats['SchoolClosureFlag'] * val_feats['school_effect_vol_365']).values.copy()
pred_b[val_feats['IsOpen'] == 0] = 0.0
pred_b = np.clip(pred_b, 0, None)

rmsle_b = calc_rmsle(y_val_true, pred_b)
wape_b = calc_wape(y_val_true, pred_b)
print(f"Model (b) - Simple Additive: RMSLE = {rmsle_b:.5f} | Accuracy = {(1 - wape_b)*100:.2f}%")

# -------------------------------------------------------------
# STEP 6.c: Model (c) - Gradient Boosting (LightGBM)
# -------------------------------------------------------------
print("\n--- Evaluating Model (c): LightGBM Regressor ---")
params_lgb = {
    'objective': 'rmse',
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 127,
    'min_child_samples': 25,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42,
    'verbose': -1,
    'n_jobs': -1
}

dtrain = lgb.Dataset(X_train, label=y_train_log)
dval = lgb.Dataset(X_val_open, label=y_val_open_log, reference=dtrain)

bst = lgb.train(
    params_lgb,
    dtrain,
    num_boost_round=1200,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

val_pred_log = bst.predict(X_val_all)
pred_c = np.expm1(val_pred_log)
pred_c[val_feats['IsOpen'] == 0] = 0.0
pred_c = np.clip(pred_c, 0, None)

rmsle_c = calc_rmsle(y_val_true, pred_c)
wape_c = calc_wape(y_val_true, pred_c)
print(f"Model (c) - LightGBM:        RMSLE = {rmsle_c:.5f} | Accuracy = {(1 - wape_c)*100:.2f}% (best iteration: {bst.best_iteration})")

# Noise Floor Analysis
open_indices = np.where(val_feats['IsOpen'] == 1)[0]
log_residuals_open = np.log1p(y_val_true[open_indices]) - np.log1p(pred_c[open_indices])
noise_floor = np.std(log_residuals_open)
print(f"Evidence Noise Floor (Std Dev of Log Residuals on Open Days): {noise_floor:.5f}")

# Multi-seed Stability Check
print("\nVerifying multi-seed stability (Seeds: 42, 123, 999)...")
seed_scores = [rmsle_c]
seed_preds = [pred_c]
for s in [123, 999]:
    p_copy = params_lgb.copy()
    p_copy['seed'] = s
    bst_s = lgb.train(
        p_copy, dtrain, num_boost_round=bst.best_iteration,
        valid_sets=[dtrain, dval], callbacks=[lgb.log_evaluation(0)]
    )
    p_log = bst_s.predict(X_val_all)
    p_val = np.expm1(p_log)
    p_val[val_feats['IsOpen'] == 0] = 0.0
    p_val = np.clip(p_val, 0, None)
    s_score = calc_rmsle(y_val_true, p_val)
    seed_scores.append(s_score)
    seed_preds.append(p_val)
    print(f"  Seed {s} Validation RMSLE: {s_score:.5f}")

pred_c_multiseed = np.mean(seed_preds, axis=0)
rmsle_multiseed = calc_rmsle(y_val_true, pred_c_multiseed)
print(f"Multi-seed Average RMSLE: {rmsle_multiseed:.5f}")

# -------------------------------------------------------------
# STEP 6.d: Model (d) - Weighted Blend
# -------------------------------------------------------------
print("\n--- Evaluating Model (d): Blend of Simple Model + GBDT ---")
best_w = 1.0
best_rmsle = 999.0
weights = np.linspace(0, 1, 101)
for w in weights:
    blended = w * pred_c_multiseed + (1.0 - w) * pred_b
    blended[val_feats['IsOpen'] == 0] = 0.0
    score = calc_rmsle(y_val_true, blended)
    if score < best_rmsle:
        best_rmsle = score
        best_w = w

pred_d = best_w * pred_c_multiseed + (1.0 - best_w) * pred_b
pred_d[val_feats['IsOpen'] == 0] = 0.0
pred_d = np.clip(pred_d, 0, None)
wape_d = calc_wape(y_val_true, pred_d)

print(f"Model (d) - Optimal Blend ({best_w*100:.1f}% GBDT + {(1-best_w)*100:.1f}% Simple): RMSLE = {best_rmsle:.5f} | Accuracy = {(1 - wape_d)*100:.2f}%")

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': bst.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
plt.title('Top 20 Features by Information Gain (LightGBM)')
plt.tight_layout()
plt.savefig('plots/6_feature_importance.png', dpi=150)
plt.close()
print("Feature importance plot saved to ./plots/6_feature_importance.png")

# -------------------------------------------------------------
# STEP 7: Full Retraining & Test Inference
# -------------------------------------------------------------
print("\n--- STEP 7: Retraining on Full Dataset (Including 2015-05-09 Window) ---")
# Include validation window in the training data for final test inference
full_train_df = pd.concat([train_df, val_feats], ignore_index=True)
full_train_open = full_train_df[(full_train_df['IsOpen'] == 1) & (full_train_df['OrderVolume'] > 0)].copy()

X_full = full_train_open[feature_cols]
y_full_log = np.log1p(full_train_open['OrderVolume'])

print(f"Full Retraining Samples: {len(X_full)} open hub-days")
dfull = lgb.Dataset(X_full, label=y_full_log)

test_cutoff = pd.Timestamp('2015-06-20')
print(f"Extracting test features as of test cutoff {test_cutoff.date()}...")
test_feats = extract_features(test, train, meta_proc, test_cutoff)
X_test = test_feats[feature_cols]

# Compute Model (b) on test
test_pred_b = (test_feats['hub_weekday_mean_vol'] + 
               test_feats['PromoActive'] * test_feats['promo_effect_vol_365'] + 
               test_feats['SchoolClosureFlag'] * test_feats['school_effect_vol_365']).values.copy()
test_pred_b[test_feats['IsOpen'] == 0] = 0.0
test_pred_b = np.clip(test_pred_b, 0, None)

# Multi-seed final prediction
final_preds_seeds = []
num_rounds = int(bst.best_iteration * 1.1)  # scale slightly for larger dataset
for s in [42, 123, 999]:
    print(f"  Training seed {s} on full dataset for {num_rounds} rounds...")
    p = params_lgb.copy()
    p['seed'] = s
    bst_full = lgb.train(p, dfull, num_boost_round=num_rounds, callbacks=[lgb.log_evaluation(0)])
    p_log = bst_full.predict(X_test)
    p_vol = np.expm1(p_log)
    p_vol[test_feats['IsOpen'] == 0] = 0.0
    p_vol = np.clip(p_vol, 0, None)
    final_preds_seeds.append(p_vol)

final_pred_gbdt = np.mean(final_preds_seeds, axis=0)

# Final ensemble
final_submission_pred = best_w * final_pred_gbdt + (1.0 - best_w) * test_pred_b
final_submission_pred[test_feats['IsOpen'] == 0] = 0.0
final_submission_pred = np.clip(final_submission_pred, 0, None)

# -------------------------------------------------------------
# STEP 8: Final Submission & Integrity Checks
# -------------------------------------------------------------
print("\n--- STEP 8: Writing and Validating submission.csv ---")
submission = pd.DataFrame({
    'Id': test['Id'],
    'OrderVolume': final_submission_pred
})

submission.to_csv('submission.csv', index=False)
print("Saved submission.csv successfully!")

# Checks
print("\n--- SUBMISSION AUDIT REPORT ---")
print(f"1. Exact row count matches (46,830 rows): {len(submission) == 46830} ({len(submission)} rows)")
print(f"2. Ids match sample_submission.csv 100%: {(submission['Id'].values == sample['Id'].values).all()}")
print(f"3. Null / NaN count: {submission.isnull().sum().sum()}")
print(f"4. Negative value count: {(submission['OrderVolume'] < 0).sum()}")
print(f"5. Exactly 0 for closed days (6,548 expected): {((test['IsOpen'] == 0) & (submission['OrderVolume'] == 0.0)).sum()}")
print(f"6. Mean predicted volume on open days: {submission[test['IsOpen'] == 1]['OrderVolume'].mean():.2f}")
print(f"7. Max predicted volume: {submission['OrderVolume'].max():.2f}")
print(f"8. Min predicted volume: {submission['OrderVolume'].min():.2f}")

print("\n" + "="*70)
print("MODEL PROGRESSION SUMMARY")
print("="*70)
print(f"Model (a) - Dumb Baseline:         RMSLE = {rmsle_a:.5f} | Acc = {(1 - wape_a)*100:.2f}%")
print(f"Model (b) - Simple Additive Model: RMSLE = {rmsle_b:.5f} | Acc = {(1 - wape_b)*100:.2f}%")
print(f"Model (c) - Tuned LightGBM (GBDT): RMSLE = {rmsle_c:.5f} | Acc = {(1 - wape_c)*100:.2f}%")
print(f"Model (d) - Multi-Seed Blend:      RMSLE = {best_rmsle:.5f} | Acc = {(1 - wape_d)*100:.2f}%")
print(f"Day-to-Day Noise Floor:            RMSLE = {noise_floor:.5f}")
print("="*70)
