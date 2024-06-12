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
2. **[Data on Danish Inflation](Danish_Inflation.csv)**: Danish Inflation.csv (*Danmarks Statistik, Statistikbanken: PRIS8*, Link: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440, Received 04/05-2024)
3. **[Data on Eurozone Inflation](Eurozone_Inflation.csv)**: Eurozone_Inflation.csv (*European Central Bank, ECB Data Portal* Link: https://data.ecb.europa.eu/publications/macroeconomic-and-sectoral-statistics/3030627, Received 05/06-2024)

**Structure of the notebook**
---

We *structure* our data project in this following way:
- **[Table of contents](dataproject.ipynb#tableofcontents):**
- **[Introduction](dataproject.ipynb#introduction):** We mention our research question of interest, we motivate it, and we mention our results.
- **[Research question](dataproject.ipynb#researchquestion):**
- **[Data analysis](dataproject.ipynb#dataanalysis):**
- **[Import of packages](dataproject.ipynb#imports):** We start by importing the packages used in the notebook (listed under Dependencies).
- **[]():**
- **[Import of the first dataset](dataproject.ipynb#firstimport):** We import the first dataset using an API.
- **[Import of the second dataset](dataproject.ipynb#secondimport):** We import the second dataset using a csv-file and Pandas.
- **[Merging of the two datasets](dataproject.ipynb#merge):** We merge the two datasets using an outer merge.
- **[Descriptive statistics](dataproject.ipynb#descriptivestatistics):** We make descriptive statistics on the merged raw data.
- **[Eurozone Inflation and the Exchange Rates](dataproject.ipynb#):** 
- **[Conclusion](dataproject.ipynb#conclusion):** We conclude our findings in the data project based on the research question.

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
