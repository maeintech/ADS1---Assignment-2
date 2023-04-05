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
    plt.title(title, fontdict={'fontsize': 20}, pad=(10))  # graph title

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
emission_upper_middle = emission_upper_middle.rename({'Russian Federation':
                                                      'Russia'})

# Plotting a bar graph for the top 5 countries
emission_upper_middle.plot(kind='bar',
                           figsize=[6, 4],
                           ylabel='CO2 emissions',
                           color=["purple",
                                  "limegreen",
                                  "crimson",
                                  "c",
                                  "orange",
                                  "darkslategray"],
                           rot=0)

# specify the title of the graph
plt.title(
    'Largest producers of CO2 emissions among Upper Middle Income Countries',
    fontdict={'fontsize': 10},
    pad=(20))

plt.savefig('Bar Plot1')  # save the chart
plt.show()  # display the chart

# to explore the GNI per capita for each of the Upper Middle Income countries
# considered, we would an excel file containing GNI per capita data using
# the previously defined read function
gni, gni_t = read('API_NY.GNP.PCAP.CD_DS2_en_excel_v2_5358755.xls', 3, 0)
gni = gni[gni['Country Name'].isin(['China',
                                    'Russian Federation',
                                    'Mexico',
                                    'South Africa',
                                    'Brazil'])]

# drop unnecessary columns
gni = gni.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
gni = gni.set_index('Country Name')  # set index to country name

# extract data for 5 year intervals between 1994 and 2019
gni = gni.loc[:, '1994':'2019':5]
gni = gni.sort_values('2019', ascending=False)
print(gni)

# creating a dataframe that shows the Population statistics for countries in
# the Upper Middle Income group
population_upper_middle = c[c['Indicator Name'] == 'Population, total']

# drop unnecessary columns and empty fields
population_upper_middle = population_upper_middle.drop(['Country Code',
                                                        'Indicator Code',
                                                        'Indicator Name'],
                                                       axis=1)

population_upper_middle = population_upper_middle.drop(
                            population_upper_middle.loc[:, '1960':'1989'],
                            axis=1)

# Renaming Russia
population_upper_middle = population_upper_middle.replace({
                                            'Russian Federation': 'Russia'})

# sorting the dataframe to get the countries producing the most CO2 emissions
population_upper_middle = population_upper_middle[population_upper_middle[
                            "Country Name"].isin(emission_upper_middle.index)]

population_upper_middle = population_upper_middle.set_index('Country Name')

# limiting the data to give only population data for 5 year intervals from
# 1994 to 2019
population_upper_middle = population_upper_middle.loc[:, '1994':'2019':5]
population_upper_middle = population_upper_middle.sort_values('2019',
                                                              ascending=False)


# Plotting a bar graph for the top 5 countries
population_upper_middle.plot(kind='bar',
                             figsize=[6, 4],
                             ylabel='Population',
                             color=["purple",
                                    "limegreen",
                                    "crimson",
                                    "c",
                                    "orange",
                                    "darkslategray"],
                             rot=0)

# specify the title of the graph
plt.title(
    'Population of Upper-middle-income Countries',
    fontdict={'fontsize': 10},
    pad=(20))

plt.savefig('Bar Plot2')  # save the chart
plt.show()  # display the chart

# To check the correlation between indicators in China, we would create
# a dataframe for China and extract indicators of interest
china = data[data['Country Name'].isin(['China'])]

# drop unnecessary columns
china = china.drop(['Country Name', 'Country Code', 'Indicator Code'], axis=1)

# extract indicators of interest
china = china[china['Indicator Name'].isin([
    "Access to electricity (% of population)",
    "Urban population",
    "Population, total",
    "Arable land (% of land area)",
    "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)",
    "Mortality rate, under-5 (per 1,000 live births)",
    "CO2 emissions (kt)",
    "Forest area (% of land area)"])]

# Replace some values to have better visualisation
china = china.replace({
    "Mortality rate, under-5 (per 1,000 live births)":
    "Infant mortality rate",
    "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":
    "Poverty headcount ratio"})

# set index to Country Name
china = china.set_index('Indicator Name')

# drop years with no values
china = china.drop(china.loc[:, '1960':'1999'], axis=1)
china = china.transpose()  # final China dataframe

# Correlation between indicators in China
china_cor = china.corr().round(2)

# plotting a heatmap to visualise correlation of indicators in China
plt.imshow(china_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(china_cor)), china_cor.columns, rotation=90)
plt.yticks(range(len(china_cor)), china_cor.columns)
plt.gcf().set_size_inches(8, 5)

# labelling the heatmap and creating legend
labels = china_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x, y, '{:.2f}'.format(labels[y, x]),
                 ha='center',
                 va='center',
                 color='black')

plt.title('Correlation Map of Indicators for China')
plt.savefig('Heat Map 1.png', bbox_inches='tight')

# To explore trends in electricity production in China, we create a new
# dataframe containing indicators for electricity production in China
china_electricity = data[data['Country Name'].isin(['China'])]

# drop unnecessary columns
china_electricity = china_electricity.drop(['Country Name',
                                            'Country Code',
                                            'Indicator Code'], axis=1)

# extract indicators of interest
ind = [
    "Electricity production from renewable sources, excluding hydroelectric (% of total)",
    "Electricity production from oil sources (% of total)",
    "Electricity production from nuclear sources (% of total)",
    "Electricity production from natural gas sources (% of total)",
    "Electricity production from hydroelectric sources (% of total)",
    "Electricity production from coal sources (% of total)"]
china_electricity = china_electricity[
    china_electricity['Indicator Name'].isin(ind)]

# Replace some values to have better visualisation
china_electricity = china_electricity.replace({
    "Electricity production from renewable sources, excluding hydroelectric (% of total)":
    "Renewable, exc hydroelectric",
    "Electricity production from oil sources (% of total)": "Oil",
    "Electricity production from nuclear sources (% of total)": "Nuclear",
    "Electricity production from natural gas sources (% of total)":
        "Natural Gas",
    "Electricity production from hydroelectric sources (% of total)":
        "Hydroelectric",
    "Electricity production from coal sources (% of total)": "Coal"})

# extract timeframe of interest
china_electricity = china_electricity.drop(
    china_electricity.loc[:, '1960':'1989'], axis=1)
china_electricity = china_electricity.set_index('Indicator Name')
china_electricity = china_electricity.loc[:, '1990':'2015':5]
china_electricity = china_electricity.fillna(0)
china_electricity.plot(kind='bar',
                       figsize=[10, 4],
                       title='Sources of Electricity production in China',
                       xlabel="Sources of electricity production",
                       ylabel='Percentage of total electricity produced',
                       color=["purple",
                              'limegreen',
                              'crimson',
                              "c",
                              "orange",
                              "darkslategray"],
                       rot=90)

plt.savefig('Bar plot3', bbox_inches='tight')


# To check the correlation between indicators in Russia, we would create
# a dataframe for Russia and extract indicators of interest
russia = data[data['Country Name'].isin(['Russian Federation'])]

# drop unnecessary columns
russia = russia.drop(['Country Name', 'Country Code', 'Indicator Code'],
                     axis=1)

# extract indicators of interest
ind2 = ['CO2 emissions (kt)',
        "Electricity production from renewable sources, excluding hydroelectric (% of total)",
        "Electricity production from oil sources (% of total)",
        "Electricity production from nuclear sources (% of total)",
        "Electricity production from natural gas sources (% of total)",
        "Electricity production from hydroelectric sources (% of total)",
        "Electricity production from coal sources (% of total)"]

russia = russia[russia['Indicator Name'].isin(ind2)]

# Replace some values to have better visualisation
russia = russia.replace({
    "Electricity production from renewable sources, excluding hydroelectric (% of total)":
    "Renewable, exc hydroelectric",
    "Electricity production from oil sources (% of total)": "Oil",
    "Electricity production from nuclear sources (% of total)": "Nuclear",
    "Electricity production from natural gas sources (% of total)":
        "Natural Gas",
    "Electricity production from hydroelectric sources (% of total)":
        "Hydroelectric",
    "Electricity production from coal sources (% of total)": "Coal"})

# extract timeframe of interest
russia = russia.drop(russia.loc[:, '1960':'1989'], axis=1)
russia = russia.drop(russia.loc[:, '2016':'2021'], axis=1)
russia = russia.set_index('Indicator Name')

russia = russia.drop('CO2 emissions (kt)', axis=0)
russia = russia.loc[:, '1990':'2015':5]

russia.plot(kind='bar',
                       figsize=[10, 4],
                       title='Sources of Electricity production in Russia',
                       xlabel="Sources of electricity production",
                       ylabel='Percentage of total electricity produced',
                       color=["purple",
                              'limegreen',
                              'crimson',
                              "c",
                              "orange",
                              "darkslategray"],
                       rot=90)

plt.savefig('Bar plot4', bbox_inches='tight')
plt.show()
             
natural = c[(c['Indicator Name'] == ("Electricity production from natural gas sources (% of total)"))]
natural = natural[natural['Country Name'].isin(['China',
                                                'Russian Federation',
                                                'Brazil',
                                                'Mexico',
                                                'South Africa'])]

natural = natural.drop(['Country Code', 'Indicator Code',
                        'Indicator Name'],axis=1)
natural = natural.set_index('Country Name')
natural = natural.drop(natural.loc[:, '1960':'1989'], axis=1)
natural = natural.drop(natural.loc[:, '2016':'2021'], axis=1)
x1 = natural.loc['Brazil',:]
x2 = natural.loc['China',:]
x3 = natural.loc['Mexico',:]
x4 = natural.loc['Russian Federation',:]
x5 = natural.loc['South Africa',:]
x = natural.columns.astype(int)
y = (x1, x2, x3, x4, x5)
plt.plot(x, x1, label = 'Brazil', linestyle='--')
plt.plot(x, x2, label = 'China', linestyle='--')
plt.plot(x, x3, label = 'Mexico', linestyle='--')
plt.plot(x, x4, label = 'Russia', linestyle='--')
plt.plot(x, x5, label = 'South Africa', linestyle='--')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Electricity production from natural gas sources (% of total)',
           fontsize=7)
plt.title('Electricity production from Natural Gas Sources')
plt.savefig('Line Plot 2')