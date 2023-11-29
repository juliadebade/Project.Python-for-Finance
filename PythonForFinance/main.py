import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go

# Charger les données
df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_04.05.2003-15.04.2023.csv")

# Filtrer les lignes avec un volume non nul
df = df[df['volume'] != 0]
df.reset_index(drop=True, inplace=True)

# Calculer RSI et EMA
df['RSI'] = ta.rsi(df.close, length=12)
df['EMA'] = ta.ema(df.close, length=150)

# Tronquer le DataFrame aux 500 premières lignes
df = df.head(5000)

# Détection de tendance
EMAsignal = [0] * len(df)
backcandles = 15

for row in range(backcandles, len(df)):
    print('.')
    uptrend = 1
    downtrend = 1
    for i in range(row - backcandles, row + 1):
        for i in range(row - backcandles, row + 1):
            if max(df.open[i], df.close[i]) >= df['EMA'][i]:
                downtrend = 0
            if min(df.open[i], df.close[i]) <= df['EMA'][i]:
                uptrend = 0
    if uptrend == 1 and downtrend == 1:
        EMAsignal[row] = 3
    elif uptrend == 1:
        EMAsignal[row] = 2
    elif downtrend == 1:
        EMAsignal[row] = 1

# Ajouter la colonne EMASignal au DataFrame
df['EMASignal'] = EMAsignal

# Fonction de détection de point pivot
def isPivot(candle, window):
    if candle - window < 0 or candle + window >= len(df):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].low > df.iloc[i].low:
            pivotLow = 0
        if df.iloc[candle].high < df.iloc[i].high:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

# Fenêtre de détection de point pivot
window_pivot = 5
df['isPivot'] = df.apply(lambda x: isPivot(x.name, window_pivot), axis=1)

# Fonction pour déterminer la position du point pivot
def pointpos(x):
    if x['isPivot'] == 2:
        return x['low'] - 1e-3
    elif x['isPivot'] == 1:
        return x['high'] + 1e-3
    else:
        return np.nan

# Ajouter la colonne pointpos au DataFrame
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)


# Afficher le graph
dfpl = df[4700:4850]
fig = go.Figure(data=[
    go.Candlestick(x=dfpl.index,
                   open=dfpl['open'],
                   high=dfpl['high'],
                   low=dfpl['low'],
                   close=dfpl['close'],
                   name='Trace 0',
                   increasing_line_color="green",
                   decreasing_line_color="red",
                   line=dict(width=1)),
    go.Scatter(x=dfpl.index,
               y=dfpl['pointpos'],
               mode='markers',
               marker=dict(size=5, color="mediumPurple"),
               name='Pivot')
])

# Ajuster les marges pour l'axe des x et y
x_start = 4700
x_margin = 10
y_margin = 0.01

# Mettre à jour la mise en page de la figure
fig.update_layout(xaxis=dict(range=[x_start - x_margin, 4850]),
                  yaxis=dict(range=[1.24 - y_margin, 1.28]),
                  xaxis_rangeslider_visible=False)

# Afficher la figure
fig.show()

print('===================================================')

# Fonction de détection de structure/motif
def detect_structure(candle, backcandles, window):
    localdf = df[candle - backcandles - window:candle - window]
    highs = localdf[localdf['isPivot'] == 1].high.tail(3).values
    idxhighs = localdf[localdf['isPivot'] == 1].high.tail(3).index
    lows = localdf[localdf['isPivot'] == 2].low.tail(3).values
    idxlowx = localdf[localdf['isPivot'] == 2].low.tail(3).index

    pattern_detected = False

    lim1 = 0.005
    lim2 = lim1 / 3
    if len(highs) == 3 and len(lows) == 3:
        order_condition = (idxlowx[0] < idxhighs[0]
                           < idxlowx[1] < idxhighs[1]
                           < idxlowx[2] < idxhighs[2])
        diff_condition = (
            abs(lows[0] - highs[0]) > lim1 and
            abs(highs[0] - lows[1]) > lim2 and
            abs(highs[1] - lows[1]) > lim1 and
            abs(highs[1] - lows[2]) > lim2
        )
        pattern_1 = (lows[0] < highs[0] and
                     lows[1] > lows[0] and lows[1] < highs[0] and
                     highs[1] > highs[0] and
                     lows[2] > lows[1] and lows[2] < highs[1] and
                     highs[2] < highs[1] and highs[2] > lows[2]
                     )

        pattern_2 = (lows[0] < highs[0] and
                     lows[1] > lows[0] and lows[1] < highs[0] and
                     highs[1] > highs[0] and
                     lows[2] < lows[1] and
                     highs[2] < highs[1]
                     )

        if (order_condition and
                diff_condition and
                (pattern_1 or pattern_2)
        ):
            pattern_detected = True

    if pattern_detected:
        return 1
    else:
        return 0

# Fenêtre pour la détection de structure
window_structure = 6
df['pattern_detected'] = df.index.map(lambda x: detect_structure(x, backcandles=40, window=window_structure))

# Afficher les lignes où un motif est détecté
columns_to_display = ['Gmt time', 'volume', 'RSI', 'EMA']
print(df[df['pattern_detected'] != 0][columns_to_display])
