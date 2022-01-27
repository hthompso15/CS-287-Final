# # plotting

# import
import pandas as pd
import os
import numpy as np
os.system('pip install statannot')
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
os.system('pip install joypy')
import joypy
import matplotlib

# directory
directory = 'stats'
diet_demo_age20plus_merged = pd.read_csv(os.path.join(directory,"diet_demo_age20plus_merged.csv"), index_col=0)
destination = 'plot'
if not os.path.exists(destination):
    os.mkdir(destination)

# HEAT MAP
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(diet_demo_age20plus_merged.corr(),  vmin=-1, vmax=1, center=0, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig(os.path.join(destination, 'heatmap.png'), dpi=300, bbox_inches='tight')

# PAIR PLOT
plt.figure(figsize=(24, 24))
selected_cols = diet_demo_age20plus_merged[['Shannon_diet', 'IDDS_10', 'SDDSRVYR', 'RIAGENDR', 'RIDAGEYR','INDFMPIR']]
sns.pairplot(selected_cols, hue='SDDSRVYR')
plt.savefig(os.path.join(destination,'pairplot.png'), dpi=300, bbox_inches='tight')

sns.set_style("whitegrid")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
#ax.set_xscale("log")

# Load the example planets dataset
#planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
sns.boxplot(x="SDDSRVYR", y="Shannon_flavor", data=diet_demo_age20plus_merged, width=.6, palette="viridis")

# Add in points to show each observation
#sns.stripplot(x="SDDSRVYR", y="Shannon_diet", data=diet_demo_age20plus_merged,
#              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.yaxis.grid(True)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Shannon Entropy",fontsize=12)
#ax.set_title("Trend of Flavor Diversity 2001-2018",fontsize=20, loc='left')
sns.despine(left=True)

add_stat_annotation(ax, data=diet_demo_age20plus_merged, x="SDDSRVYR", y="Shannon_flavor",
                    box_pairs=[(2003, 2005), (2003, 2017), (2003, 2009)],
                    test='t-test_ind', text_format='star', loc='outside', verbose=2)
plt.savefig(os.path.join(destination,'boxplot.png'), dpi=300, bbox_inches='tight')


# ECDF
plt.figure(figsize=(10, 6))
ax = sns.ecdfplot(
    data=diet_demo_age20plus_merged,
    x="Shannon_flavor", hue="SDDSRVYR", palette="viridis")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Shannon Entropy",fontsize=12)
plt.savefig(os.path.join(destination,'ECDF.png'), dpi=300, bbox_inches='tight')

# VIOLIN
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x="SDDSRVYR", y="Shannon_diet",
                    data=diet_demo_age20plus_merged,
                    scale="width", palette="viridis")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Shannon Entropy",fontsize=12)
plt.savefig(os.path.join(destination,'violin.png'), dpi=300, bbox_inches='tight')

# SCATTER PLOT
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="INDFMPIR", y="Shannon_diet",
                hue="berry_crop_6yr_prior",
                # size="depth",
                sizes=(1, 8), linewidth=0,
                palette="viridis",
                data=diet_demo_age20plus_merged, alpha=0.33, s=10, ax=ax)
ax.set_ylabel("Diet Diversity (Shannon)", fontsize=12)
ax.set_xlabel("Income Poverty Ratio",fontsize=12)
plt.savefig(os.path.join(destination,'scatter.png'), dpi=300, bbox_inches='tight')

# Joyplot

plt.figure(figsize=(10.5,4), dpi= 200)

norm = plt.Normalize(diet_demo_age20plus_merged["berry_crop_6yr_prior"].min(), diet_demo_age20plus_merged["berry_crop_6yr_prior"].max())
ar = np.array(diet_demo_age20plus_merged["berry_crop_6yr_prior"])

original_cmap = plt.cm.cividis
cmap = matplotlib.colors.ListedColormap(original_cmap(norm(ar)))
sm = matplotlib.cm.ScalarMappable(cmap=original_cmap, norm=norm)
sm.set_array([])

fig, axes = joypy.joyplot(diet_demo_age20plus_merged, by="SDDSRVYR", column="Shannon_diet", figsize=(10.5,4),colormap = cmap)
fig.colorbar(sm, ax=axes, label="Crop Diversity")
plt.xlabel("Dietary Diversity")

# Decoration
#plt.title('Trend of Dietary Diversity', fontsize=22)
plt.savefig(os.path.join(destination,'joyplot.png'), dpi=300, bbox_inches='tight')
