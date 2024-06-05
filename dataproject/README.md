# Data project

**Group members:**
---

- [Sheng Ye Michael Chen (nsr708)](https://github.com/nsr708), Exercise Class 6
- [Anton Meier Ebsen Jørgensen (lpx972)](https://github.com/AntonEbsen), Exercise Class 1

**Introduction:**
---

Our project is titled **EXCHANGE RATE DYNAMICS: THE RELATIONSHIP BETWEEN DANISH INFLATION AND RELATIVE EXCHANGE RATES**, and it is about how danish inflation affects the exchange rates of the US Dollar and the Euro.

**Data sets:**
---

We apply the **following datasets** to answer our research question:

1. **[Data on Exhange Rates](dataVA.xlsx)**: dataVA.xlsx (*Danmarks Statistik, Statistikbanken: DNVALA, Link: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440, 03/04-2024*)
2. **Data on Danish Inflation**: Inflation.csv (*Danmarks Statistik, Statistikbanken: PRIS8, Link: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440, 04/05-2024*)
3. **Data on Eurozone Inflation**: 

**Structure of the notebook:**
---

We *structure* our data project in this following way:
- **Introduction:** We mention our research question of interest, we motivate it, and we mention our results. 
- **Import of packages:** We start by importing the packages used in the notebook (listed under Dependencies).  
- **Import of the first dataset:** We import the first dataset using an API.
- **Merging the two data sets:** We merge the two datasets.
- **Descriptive statistics:** We make descriptive statistics on the merged raw data.
- **Conclusion:** We conclude our findings in the data project.

**Instructions:**
---

The **results** of this data project can be seen from running the notebook: [dataproject.ipynb](dataproject.ipynb).

**Dependencies:** 
---

Apart from a standard Anaconda Python 3 installation, this data project requires installation of the API and the following packages: 
- **dstapi**: We use the API created by Alessandro Martinello to import the data from Statistikbanken on the Exchange Rates. (Link to the repo: https://github.com/alemartinello/dstapi).
- **Matplotlib.pyplot**: We use the Matplotlib.pyplot-package to make graphs of the raw data.
- **Pandas**: We use the Pandas-package for cleaning, storing, and processing the data. 
- **dataproject.py**: We use the dataproject.py-file to have the classes for our codes. This makes the notebook more neat.
   
