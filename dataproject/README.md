# Data project
**Group members**
---

- [Sheng Ye Michael Chen (nsr708)](https://github.com/nsr708), Exercise Class 6
- [Anton Meier Ebsen Jørgensen (lpx972)](https://github.com/AntonEbsen), Exercise Class 1

**Introduction**
---

Our project is titled **EXCHANGE RATE DYNAMICS: THE RELATIONSHIP BETWEEN DANISH INFLATION AND EXCHANGE RATES**, and it is about how danish inflation affects the exchange rates of the US Dollar and the Euro.

**Data sets**
---

We apply the **following datasets** to answer our research question:

1. **[Data on Exhange Rates](dataVA.xlsx)**: dataVA.xlsx (*Danmarks Statistik, Statistikbanken: DNVALA*, Link: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440, Received 03/04-2024)
2. **[Data on Danish Inflation (CPI)](Danish_Inflation.csv)**: Danish Inflation.csv (*Danmarks Statistik, Statistikbanken: PRIS8*, Link: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440, Received 04/05-2024)
3. **[Data on Eurozone Inflation (HICP)](Eurozone_Inflation.csv)**: Eurozone_Inflation.csv (*European Central Bank, ECB Data Portal* Link: https://data.ecb.europa.eu/publications/macroeconomic-and-sectoral-statistics/3030627, Received 05/06-2024)

**Structure of the notebook**
---

We *structure* our data project in this following way:
- **[Table of contents](dataproject.ipynb#tableofcontents)**
1. **[Introduction](dataproject.ipynb#introduction)**
2. **[Research question](dataproject.ipynb#researchquestion)**
3. **[Data analysis](dataproject.ipynb#dataanalysis)**
4. **[Read and clean the two datasets](dataproject.ipynb#readandclean)**
5. **[The first dataset: The three exchange rates (EUR, USD, and CNY)](dataproject.ipynb#importexchangerate)**
6. **[The second dataset: Danish Inflation (Consumer Price Index)](dataproject.ipynb#importinflation)**
7. **[Merging of the two datasets](dataproject.ipynb#merging)**
8. **[Descriptive statistics of the merged dataset](dataproject.ipynb#dsmerged)**
9. **[Descriptive statistics](dataproject.ipynb#descriptivestatistics)**
10. **[Descriptive analysis of the three exchange rates](dataproject.ipynb#theexchangerates)**
11. **[Descriptive analysis of the exchange rates and Danish inflation](#dadanishinflation)**
12. **[Descriptive analysis of the exchange rates and Eurozone inflation](dataproject.ipynb#eurozoneinflation)**
13. **[Conclusion](dataproject.ipynb#conclusion)**

**Instructions**
---

The **results** of this data project can be seen from running the notebook: [dataproject.ipynb](dataproject.ipynb).

**Dependencies** 
---

Apart from a standard Anaconda Python 3 installation, this data project requires installation of the API and the following packages: 
- **dstapi**: We use the API created by Alessandro Martinello to import the data from Statistikbanken on the Exchange Rates. (Link to the repo: https://github.com/alemartinello/dstapi).
- **Matplotlib.pyplot**: We use the Matplotlib.pyplot-package to make graphs of the raw data.
- **Pandas**: We use the Pandas-package for cleaning, storing, and processing the data. 
- **dataproject.py**: We use the dataproject.py-file to have the classes for our codes. This makes the notebook more neat.
