"""
Hotel Bookings Analysis
========================
Covers:
2.1 Booking Volume & Seasonality
2.2 Impact of Stay Duration on Cancellation Rate
2.3 Impact of Lead Time on Cancellation Rate
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 110
COLORS = {'City Hotel': '#2E86AB', 'Resort Hotel': '#E67E22'}

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv('hotel_bookings_data.csv')  # <-- change path if needed

month_order = ['January','February','March','April','May','June','July',
               'August','September','October','November','December']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'],
                                           categories=month_order, ordered=True)
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_weekdays_nights']

# =================================================================
# 2.1  BOOKING VOLUME & SEASONALITY
# =================================================================

# --- Q1: Share of bookings by hotel type (pie chart) ---
share = df['hotel'].value_counts()
share_pct = df['hotel'].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(share, labels=share.index, autopct='%1.1f%%', startangle=90,
       colors=[COLORS[h] for h in share.index],
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Share of Bookings by Hotel Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_booking_share_pie.png', bbox_inches='tight')
plt.close()

print(f"City Hotel: {share_pct['City Hotel']:.1f}% | Resort Hotel: {share_pct['Resort Hotel']:.1f}%")
print(f"City Hotel is booked ~{share_pct['City Hotel'] - share_pct['Resort Hotel']:.0f} "
      f"percentage points more than Resort Hotel "
      f"(~{share_pct['City Hotel']/share_pct['Resort Hotel']:.1f}x as many bookings).")

# --- Q2: Bookings per month per hotel type (line chart) ---
monthly = df.groupby(['arrival_date_month', 'hotel'], observed=True).size().unstack()

fig, ax = plt.subplots(figsize=(10, 5))
for hotel in monthly.columns:
    ax.plot(monthly.index, monthly[hotel], marker='o', label=hotel,
            color=COLORS[hotel], linewidth=2.5)
ax.set_title('Bookings per Month by Hotel Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Bookings')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('02_monthly_bookings_line.png', bbox_inches='tight')
plt.close()

busiest = monthly.sum(axis=1).idxmax()
quietest = monthly.sum(axis=1).idxmin()
print(f"Busiest month overall: {busiest} | Quietest month overall: {quietest}")

# =================================================================
# 2.2  STAY DURATION vs CANCELLATION RATE
# =================================================================

# --- Q1: Cancellation rate by hotel type (bar chart) ---
cancel_rate_hotel = df.groupby('hotel')['is_canceled'].mean() * 100

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(cancel_rate_hotel.index, cancel_rate_hotel.values,
              color=[COLORS[h] for h in cancel_rate_hotel.index])
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
            ha='center', fontweight='bold')
ax.set_title('Cancellation Rate by Hotel Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Cancellation Rate (%)')
plt.tight_layout()
plt.savefig('03_cancel_rate_by_hotel.png', bbox_inches='tight')
plt.close()

print(cancel_rate_hotel)

# --- Q2: Cancellation rate vs total length of stay ---
# Cap stay length to avoid noisy long tail (e.g. >=15 nights grouped)
df['stay_bucket'] = df['total_stay'].clip(upper=15)

stay_cancel = (df.groupby(['stay_bucket', 'hotel'], observed=True)['is_canceled']
                 .mean().mul(100).unstack())

fig, ax = plt.subplots(figsize=(10, 5))
for hotel in stay_cancel.columns:
    ax.plot(stay_cancel.index, stay_cancel[hotel], marker='o',
            label=hotel, color=COLORS[hotel], linewidth=2)
ax.set_title('Cancellation Rate vs. Length of Stay', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Nights Stayed (15 = 15 or more)')
ax.set_ylabel('Cancellation Rate (%)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('04_cancel_rate_vs_stay_length.png', bbox_inches='tight')
plt.close()

# correlation to describe trend direction
for hotel in ['City Hotel', 'Resort Hotel']:
    sub = df[df['hotel'] == hotel]
    corr = sub['total_stay'].corr(sub['is_canceled'])
    print(f"{hotel}: correlation(total_stay, is_canceled) = {corr:.3f}")

# =================================================================
# 2.3  LEAD TIME vs CANCELLATION RATE
# =================================================================

# lead_time = number of days between the booking date and the arrival date

df['lead_time_bucket'] = pd.cut(
    df['lead_time'],
    bins=[-1, 7, 30, 60, 90, 180, 365, df['lead_time'].max()],
    labels=['0-7', '8-30', '31-60', '61-90', '91-180', '181-365', '365+']
)

lead_cancel = (df.groupby(['lead_time_bucket', 'hotel'], observed=True)['is_canceled']
                 .mean().mul(100).unstack())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(lead_cancel.index))
width = 0.35
for i, hotel in enumerate(lead_cancel.columns):
    ax.bar(x + i*width, lead_cancel[hotel], width, label=hotel, color=COLORS[hotel])
ax.set_xticks(x + width/2)
ax.set_xticklabels(lead_cancel.index)
ax.set_title('Cancellation Rate vs. Lead Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Lead Time (days before arrival)')
ax.set_ylabel('Cancellation Rate (%)')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('05_cancel_rate_vs_lead_time.png', bbox_inches='tight')
plt.close()

print(lead_cancel)

for hotel in ['City Hotel', 'Resort Hotel']:
    sub = df[df['hotel'] == hotel]
    corr = sub['lead_time'].corr(sub['is_canceled'])
    print(f"{hotel}: correlation(lead_time, is_canceled) = {corr:.3f}")

print("\nAll charts saved as PNG files in the current directory.")
