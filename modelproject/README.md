# Model analysis project

**Group members**
---

- [Sheng Ye Michael Chen (nsr708)](https://github.com/nsr708), Exercise class 6
- [Anton Meier Ebsen Jørgensen (lpx972)](https://github.com/AntonEbsen), Exercise class 1

**Introduction**
---

Our model project is titled **STEADY STATE IN THE REAL BUSINESS CYCLE MODEL** and it is about finding the steady state values numerically in the Real Business Cycle model (RBC). The code of the equations in the system are inspired by [Chad Fulton (2015)](https://github.com/ChadFulton/tsa-notebooks/blob/master/estimating_rbc.ipynb), however, we modify the technology equation and production function in the extension of the model. We make an extension of the RBC model, where the production function is a CES production function instead of the typical Cobb-Douglass production function. 

**The structure of the project**
---
We structure our model project as follows:
- **[Table of contents](modelproject.ipynb#tableofcontents)**
1. **[Introduction](modelproject.ipynb#introduction)**
2. **[Model description](modelproject.ipynb#modeldescription)**
3. **[Steady state equations](modelproject.ipynb#steadystateequations)**
4. **[Numerical solution](modelproject.ipynb#numericalsolution)**
5. **[Linear approximations](modelproject.ipynb#linearapproximations)**
6. **[Log-Linerization](modelproject.ipynb#loglinearization)**
7. **[Root finding algorithm](modelproject.ipynb#rootfindingalgorithm)**
8. **[The Log-linerized system of equations](modelproject.ipynb#loglinearsystem)**
9. **[Code](modelproject.ipynb#code)**
10. **[Static plot for the Steady State values](modelproject.ipynb#staticplot)**
11. **[Interactive plot for the Steady State values](modelproject.ipynb#interactiveplot)**
12. **[An extension of the model](modelproject.ipynb#extension)**
13. **[Conclusion](modelproject.ipynb#conclusion)**

**Instructions**
---

The **results** of the model project can be seen from running the notebook: [modelproject.ipynb](modelproject.ipynb).

**Dependencies**
---

Apart from a standard Anaconda Python 3 installation, the project requires these packages.
- **Matplotlib.pyplot**: We use the *Matplotlib.pyplot*-package to illustrate the steady state values of the RBC model. 
- **Numpy**: We use the *Numpy*-package to return the steady state values as a Numpy-array.
- **Scipy**: We use the *Scipy*-package to import optimize such that we can use the root-finding algorithm.
- **Types**: We use the *Types*-package to import SimpleNameSpace.
- **Modelproject.py**: We use the *Modelproject.py*-package to store our code instead of having it all in the notebook.
