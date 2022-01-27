# Final Project CS287
###TEAM NAME: Wireless Octopus 
###TEAM MEMBERS: Tung-Lin Liu, John Richardson, Thomas Hanlon and Harrison Thompson

## abstract
The study investigates the relationships between crop, dietary and flavor diversity using U.S supply (production and trade) and dietary data over time. The data suggests the decline of crop diversity in the last 20 years co-occurs with the decline of dietary and flavor diversity. Unlike low- and middle-income countries, income is a weak predictor of dietary diversity in the US. Alternatively, the model in this study found that flavor diversity follows the same s-curve relationship expected between income and dietary diversity in the Economics literature, therefore, provides another means to understand the complex process between food environment and human diets. ## code

### datacleaning.py

datacleaning.py converts the raw datasets into a form we can run analysis on. It merges information from all of our datasets
and writes the relevant data to csv files <br> <br>
In particular, it:
<ol>
    <li>Loads US-specific data from GENuS</li>
    <li>Loads lookup tables for FCID codes and food names</li>
    <li>Converts meal names to ingredient weights</li>
    <li>Clusters ingredients into categories for analysis</li>
    <li>Matches ingredients with FlavorDB data</li>
    <li>Writes out resulting dataframes to CSVs</li>
</ol>


### EDA.py
EDA.py contains our exploratory data analysis <br>
The process is broken down into the following stages:
<ul>
    <li>EDA for demographic data</li>
    <li>EDA for Diet data</li>
    <li>EDA for UNFAO-IDDS</li>
    <li>EDA for FlavorDV</li>
    <li>Final EDA Plotting</li>
</ul>

### analysis.py
analysis.py is where the bulk of the statistical calculations occur.<br>
The two models developed are as follows:
<ul>
    <li>Diet Diversity (Shannon Entropy) vs. Diet Diversity (Berry) and Income</li>
    <li>Diet Diversity (Shannon) vs. Demographic Information</li>
</ul>

### plotting.py
plotting.py generates plots from the cleaned and processed dataset<br>
The types of plots developed are as follows:
<ul>
    <li>Heatmap of correlations between diversity metrics and demographic data</li>
    <li>Pair Plot of Diet data</li>
    <li>ECDFs of Shannon Entropy for multiple features</li>
    <li>Scatterplots of Diet Diversity by Income Poverty Ratio</li>
    <li>Joyplot of Diet diversity and Crop Diversity</li>
</ul>

# Ideas Along The Way
ideas.pdf contains detailed descriptions of our thoughts and processes along the development of this process <br>

## data 

### FAOSTAT data 
[from nutritional Stability Paper (https://github.com/dbemerydt/nutriStability/tree/V1.0.1)]

production.csv
Tonnes of each crop produced by each country each year. Retrieved from FAO at http://www.fao.org/faostat/en/#data/QCL/metadata.

prodPlusImp.csv
Sum of tonnes of each crop produced and imported by each country each year. Retrieved from FAO at http://www.fao.org/faostat/en/#data/QCL/metadata.

### NHANES

### FCID
[https://fcid.foodrisk.org/]

Recipes_WWEIA_FCID_0510.csv. U.S. EPA recipe database to translate WWEIA food consumption to consumption of agricultural food commodities

### FLavorDB
FlavorDB scraper [https://vchoo.github.io/]

