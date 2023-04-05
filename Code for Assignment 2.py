#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:08:48 2023

@author: Mae
"""
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define all functions which will be used in the program


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)

    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0

    return value


def line_graph(x, y):
    """ Creates a line graph of specified variables """

    # specify the figure which is to plotted
    graph = plt.figure(figsize=(10, 6))

    plt.plot(x, y1, label=label1)  # line plot for line 1
    plt.plot(x, y2, label=label2)  # line plot for line 2
    plt.plot(x, y3, label=label3)  # line plot for line 3
    plt.plot(x, y4, label=label4)  # line plot for line 4
    plt.plot(x, y5, label=label5)  # line plot for line 5

    plt.xlabel(x_label)  # label the x-axis
    plt.ylabel(y_label)  # label the y-axis
    plt.title(title, fontdict={'fontsize': 20}, pad=(10))  # title of the graph

    plt.legend()  # show a legend on the plot
    return graph


# import data from excel file
def read(x, y, z):
    """
    Reads and imports files from excel spreadsheet to a python DataFrame

    Returns two dataframes with the second dataframe being the transpose of
    the first
    """

    data = pd.read_excel(x, skiprows=y, sheet_name=z)
    data_transpose = data.transpose()
    return data, data_transpose


# specify parameters and call the function to read the excel sheet containing
# the data
x = 'API_19_DS2_en_excel_v2_5252304.xls'
y = 3
z = 0

# call the function to read the data
data, data_transpose = read(x, y, z)

# Slicing the dataframe to get data for Country groups of interest
income_groups = data[data['Country Name'].isin(['High income',
                                                'Low income',
                                                'Lower middle income',
                                                'Upper middle income',
                                                'World'])]

# creating and cleaning a new dataframe with CO2 emissions
# for Country groups of interest
CO2_emission = income_groups[(income_groups['Indicator Name'] ==
                             'CO2 emissions (kt)')]
CO2_emission = CO2_emission.round(2)  # rounding off values to 2dp

# drop unnecessary columns and empty fields
CO2_emission = CO2_emission.drop(['Country Code',
                                  'Indicator Code',
                                  'Indicator Name'], axis=1)
CO2_emission = CO2_emission.dropna(axis=1)
CO2_emission = CO2_emission.set_index('Country Name')
CO2_emission = CO2_emission.transpose()  # final dataframe for CO2 emissions

# Exploring the statistical properties for each Country group
CO2_emission.describe()
skew(CO2_emission)  # calling the previously defined skewness function
kurtosis(CO2_emission)  # calling the previously defined kurtosis function

# to plot a line graph for CO2 emissions, we need to specify parameters as
# listed in the previously defined line graph function

CO2_emission.index = pd.to_numeric(CO2_emission.index)
x = CO2_emission.index
y1 = CO2_emission['High income']
y2 = CO2_emission['Low income']
y3 = CO2_emission['Lower middle income']
y4 = CO2_emission['Upper middle income']
y5 = CO2_emission['World']

label1 = 'High income'
label2 = 'Low income'
label3 = 'Lower middle income'
label4 = 'Upper middle income'
label5 = 'World'

title = 'CO2 Emissions for Country-wide income groups'
x_label = 'Year'
y_label = 'CO2 Emissions'

# after specifying all parameters, we can now call the function
line_graph(x, y)
plt.savefig('Line Plot1')  # save the graph
plt.show()  # display the graph

# Investigate further to know which countries are in the Upper Middle income
# group by calling the read function to read another sheet in our data
# which contains metadata
metadata, metadata_t = read('API_19_DS2_en_excel_v2_5252304.xls', 0, 1)

# Extracting the countries which are in the Upper Middle Income group
upper_middle_income = metadata[metadata['IncomeGroup'].isin([
                                        'Upper middle income'])]

# creating a dataframe that shows the CO2 emissions for countries in the
# Upper Middle Income group
c = data[data['Country Name'].isin(upper_middle_income['TableName'])]
emission_upper_middle = c[(c['Indicator Name'] == 'CO2 emissions (kt)')]
emission_upper_middle = emission_upper_middle.round(2)  # rounding off to 2dp

# drop unnecessary columns and empty fields
emission_upper_middle = emission_upper_middle.drop(['Country Code',
                                                    'Indicator Code',
                                                    'Indicator Name'], axis=1)
emission_upper_middle = emission_upper_middle.drop(
                        emission_upper_middle.loc[:, '1960':'1989'], axis=1)

emission_upper_middle = emission_upper_middle.set_index('Country Name')

# sorting the dataframe to get the countries producing the most CO2 emissions
emission_upper_middle = emission_upper_middle.sort_values('2019',
                                                          ascending=False)
# limiting the data to give only the top 5 countries with their CO2 emission
# for 5 year intervals from 1994 to 2019
emission_upper_middle = emission_upper_middle[0:5]
emission_upper_middle = emission_upper_middle.loc[:, '1994':'2019':5]

# Renaming Russia for a nicer visualisation
emission_upper_middle = emission_upper_middle.rename({'Russian Federation'
                                                      'Russia'})

# Plotting a bar graph for the top 5 countries
emission_upper_middle.plot(kind='bar',
                           figsize=[6, 4],
                           ylabel='CO2 emissions',
                           color=["darkslategray",
                                  "limegreen",
                                  "crimson",
                                  "c",
                                  "orange"],
                           rot=0)

# specify the title of the graph
plt.title(
    'Largest producers of CO2 emissions among Upper Middle Income Countries',
    fontdict={'fontsize': 12},
    pad=(20))

plt.savefig('Bar Plot1')  # save the chart
plt.show()  # display the chart
