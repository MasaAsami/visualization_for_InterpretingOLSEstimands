import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


treated_ratio = (
    st.sidebar.slider("treated_ratio", min_value=1, max_value=100, step=1, value=15)
    / 100
)

sample_size = 1000
_theta = 0.15
_scale = 5


treated_n = int(sample_size * treated_ratio)
control_n = sample_size - treated_n
treated_x_mean = 15
control_x_mean = -15

np.random.seed(1)
x_treated = np.random.normal(loc=treated_x_mean, scale=10, size=treated_n)
treated = pd.DataFrame(
    {
        "x": x_treated,
    }
)

x_control = np.random.normal(loc=control_x_mean, scale=10, size=control_n)
control = pd.DataFrame(
    {
        "x": x_control,
    }
)


df = pd.concat([treated, control]).reset_index(drop=True)

df["ps"] = np.clip(1 / (1 + np.exp(-1 * df["x"] * _theta)), 0.01, 0.99)
df["latent_group"] = np.where(df["ps"] >= 0.5, 1, 0)
df["t"] = df["ps"].apply(lambda x: np.random.binomial(1, x, size=None))
df["group"] = df["t"].apply(lambda x: "treated" if x > 0 else "control")
df["y"] = df.eval("x + x*t") + np.random.normal(
    loc=0, scale=_scale * 0.2, size=sample_size
)


ols_model_treated = smf.ols(formula="y ~ x", data=df.query("t>0")).fit()
ols_model_control = smf.ols(formula="y ~ x", data=df.query("t<1")).fit()

##############
# visualization
##############

st.title("visualization for Interpreting OLS Estimands")


# 円グラフ
fig_target = go.Figure(
    data=[
        go.Pie(
            labels=["treated", "control"],
            values=[len(df.query("t>0")), len(df.query("t<1"))],
            hole=0.3,
        )
    ]
)
fig_target.update_layout(
    showlegend=False, height=200, margin={"l": 20, "r": 60, "t": 0, "b": 0}
)
fig_target.update_traces(textposition="inside", textinfo="label+percent")

st.sidebar.markdown("## treated ratio")
st.sidebar.plotly_chart(fig_target, use_container_width=True)


ols_model = smf.ols(formula="y ~ t + x", data=df).fit()
estemated_effect = ols_model.params["t"]

st.write(f"## OLS (y ~ t + x): {estemated_effect : .2f}")

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="group",
    trendline="ols",
    width=1000,
    height=800,
    marginal_x="histogram",
    opacity=0.2,
)

t1avg = df.query("t>0")["x"].mean()
ey1_t1_t1avg = ols_model_treated.params[0] + ols_model_treated.params[1] * t1avg
ey0_t0_t1avg = ols_model_control.params[0] + ols_model_control.params[1] * t1avg

t0avg = df.query("t<1")["x"].mean()
ey1_t1_t0avg = ols_model_treated.params[0] + ols_model_treated.params[1] * t0avg
ey0_t0_t0avg = ols_model_control.params[0] + ols_model_control.params[1] * t0avg

fig.add_annotation(
    x=df.query("t>0")["x"].max(),
    y=ols_model_treated.params[0]
    + ols_model_treated.params[1] * df.query("t>0")["x"].max(),
    text="E[Y(1)| T=1, X=x]",
    showarrow=True,
    arrowhead=1,
)
fig.add_annotation(
    x=df.query("t<1")["x"].min(),
    y=ols_model_control.params[0]
    + ols_model_control.params[1] * df.query("t<1")["x"].min(),
    text="E[Y(0)| T=0, X=x]",
    showarrow=True,
    arrowhead=1,
)


fig.add_trace(
    go.Scatter(
        x=[t1avg, t1avg],
        y=[ey1_t1_t1avg, ey0_t0_t1avg],
        mode="lines+markers",
        name="",
        line_color="black",
    )
)
fig.add_trace(
    go.Scatter(
        x=[t0avg, t0avg],
        y=[ey1_t1_t0avg, ey0_t0_t0avg],
        mode="lines+markers",
        name="",
        line_color="black",
    )
)

att = ey1_t1_t1avg - ey0_t0_t1avg
att = df.query("t>0")["x"].mean()
atu = ey1_t1_t0avg - ey0_t0_t0avg
atu = df.query("t<1")["x"].mean()
treated_ratio = len(df.query("t>0")) / len(df)
control_ratio = 1 - treated_ratio
var_t1_ps = df.query("t>0")["ps"].var()
var_t0_ps = df.query("t<1")["ps"].var()

_weight = (
    control_ratio * var_t0_ps / (treated_ratio * var_t1_ps + control_ratio * var_t0_ps)
)

st.write(f"### ATT : {att :.2f}")
st.write(f"### ATU : {atu :.2f}")
st.write(f"### weight : {_weight :.2f}")
st.write(f"## theoretical value : {att*_weight + atu*(1-_weight)}")

fig.add_annotation(
    x=t1avg,
    y=ey0_t0_t1avg * 0.5 + ey1_t1_t1avg * 0.5,
    xref="x",
    yref="y",
    text=f"ATT:{ey1_t1_t1avg - ey0_t0_t1avg :.2f}",
    showarrow=True,
    font=dict(family="Courier New, monospace", size=16, color="black"),
    align="left",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    ax=20,
    ay=-30,
    bordercolor="black",
    borderwidth=2,
    borderpad=4,
    bgcolor="white",
    opacity=0.5,
)
fig.add_annotation(
    x=t0avg,
    y=ey0_t0_t0avg * 0.5 + ey1_t1_t0avg * 0.5,
    xref="x",
    yref="y",
    text=f"ATU:{ey1_t1_t0avg - ey0_t0_t0avg :.2f}",
    showarrow=True,
    font=dict(family="Courier New, monospace", size=16, color="black"),
    align="left",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    ax=20,
    ay=-30,
    bordercolor="black",
    borderwidth=2,
    borderpad=4,
    bgcolor="white",
    opacity=0.5,
)


if len(df.query("t>0")) > len(df.query("t<1")):
    fig.add_vline(
        x=df.query("t>0")["x"].mean(),
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="treated average",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=df.query("t<1")["x"].mean(),
        line_width=3,
        line_dash="dash",
        line_color="blue",
        annotation_text="control average",
        annotation_position="top right",
    )
else:
    fig.add_vline(
        x=df.query("t>0")["x"].mean(),
        line_width=3,
        line_dash="dash",
        line_color="blue",
        annotation_text="treated average",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=df.query("t<1")["x"].mean(),
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="control average",
        annotation_position="top right",
    )

st.plotly_chart(fig)

st.write(df)
