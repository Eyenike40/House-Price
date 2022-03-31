from operator import index
from matplotlib import markers
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import altair as alt
import joblib 
import plotly.express as px
from PIL import Image
import numpy as np
import time



house_img = Image.open("House.png")
house =Image.open("House Image.jpg")
st.set_page_config(page_title="Housing Price", page_icon=house_img)

#Title of my website
left_column, right_column = st.columns([1.5,3.5])
with left_column:
    st.text("")
    st.image(house)

right_column.title("Price Housing App")

#Briefly describe what the App does
st.write("""
This app help to **predict** the housing price given the land area and the town
""")

#Dataframe of housing price
df_house = pd.read_csv("homeprices.csv")

if st.checkbox("Show Housing data"):
    with st.spinner('Housing Price data loading...'):
        time.sleep(5)

        #Sub-title of the App
        st.subheader("Data for House Price")
        st.dataframe(df_house.reindex(np.arange(1,11)).head(10))
        st.write("\n\n")

#Plot the area of house against the price of house
fig = px.scatter(df_house, x="area", y="price", symbol="town", color="town", labels={
    "area":"Area (sq. ft)", "price":"Price (thousand of dollar)"},
     title="Plot showing the relationship between land area and price of house"
)

fig.update_layout(
    legend_title="Town",
    font=dict(size=13.5)
)

fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey", showline=True, linewidth=0.5,
linecolor="black", mirror=True)
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey", showline=True, linewidth=0.5,
linecolor="black", mirror=True)
fig.update_traces(marker=dict(size=7.5))

st.write(fig)

desc_house = df_house.groupby("town").agg(["mean", "min", "max"])
st.write("#### Descriptive Statistic on House Variables")
st.table(desc_house,)


#Input variabe for model prediction
st.sidebar.header("Input Parameter")
area = st.sidebar.slider("Area of the house (sq. ft)", 2500, 4200, 3000)
st.write("\n\n")

town_radio = st.sidebar.radio("Select town", options=["Robinsville", "West Windsor", "Monroe Township"])

robinsville = 0
west_windsor = 0

if town_radio == "Robinsville":
    robinsville = 1
elif town_radio == "West Windsor":
    west_windsor = 1

#Display model equation
st.write("\n\n#### Model Equation and Goodness of Fit")
st.write("House Price = 209776.39 + 126.90 x land area + 25686.41 x Robinsville + 40013.98 x West Windsor")
st.write("R$^2$ = 0.9564")


#Load calibrated housing model and predict
st.write("\n\n#### Predict price of the house")
model = joblib.load("House Price Model (town)")
price = model.predict([[area, robinsville, west_windsor]])
st.write(f"Area of House = {area} ft$^2$")
st.write(f"Town = {town_radio}")
st.write(f"House price = $ {price[0]:,.2f}")

st.success("House Price Prediction Completed")

