import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import math

pio.templates.default = "ggplot2"

FINAL = "stats/final.csv"

df_final = pd.read_csv(FINAL)
df_final.head()

df_final.groupby('method').describe()['execution_time']

fig = px.box(df_final, x='instance', y='value', color='method')
go.FigureWidget(fig)
fig.show()
fig.write_image("images/fig1.png")

fig = px.box(df_final, x='instance', y='execution_time', color='method')
go.FigureWidget(fig)
fig.show()
fig.write_image("images/fig1.png")


def print_stats(df, metric):
    count_tries = df['run'].max()
    stats = df.groupby(['instance', 'method'])[metric].agg(['mean', 'std'])
    ci95_hi = []
    ci95_lo = []

    for i in stats.index:
        m, s = stats.loc[i]
        ci95_hi.append(m + 1.96*s/math.sqrt(count_tries))
        ci95_lo.append(m - 1.96*s/math.sqrt(count_tries))

    stats['ci95_lo'] = ci95_lo
    stats['ci95_hi'] = ci95_hi
    print(stats)


print_stats(df_final, 'value')

print_stats(df_final, 'execution_time')