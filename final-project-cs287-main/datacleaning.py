# TO-DO (update 2021-1-12):
# 0.  7000 recipes are from 2005-2010. for the diet data post 2010, there will be new dishes not in our recipe database
# 1.  Crop Diversity (GENuS) calculate diversity metrics (shannon entropy)
# 2.  Diet Diversity (NHANES-WWEIA): calculate diversity metrics (count-based, entropy-based)
# 2.5 Link the flat_sum_10 to demographics by 'SEQN'
# 3.  Flavor Diversity (FlavorDB): calculate diversity metrics (count-based, entropy-based)

# 4. Poly regression with knots:
# dietary diversity (from 2) ~ income brackets (from 2.5 ) + gender (from 2.5) + crop diversity of the same year (from 1)

# 5. Time-series:
# plot the decline flavor diversity (from 3)

# CODE STRUCTURE
#  X. BASIC
#  |-- imports
#  |-- local variables
#  |-- helper functions
#  |-- Data class
#  0. EXTRACT US-ONLY DATA FROM GENuS (GENuS replaces FAOSTAT)
#  1. CREATING LOOKUP TABLES
#  2. CONVERT FOOD FROM 24HR DIETARY RECALL TO FCID WEIGHTS
#  |-- 2.1. CREATE A PANDA DATAFRAME FROM  DIETS_TO_WEIGHTS MULTI-LEVEL DICTIONARY
#  3. MATCH FCID TO UNFAO-IDDS
#  4. MATCH FCID TO FlavorDB

### imports ###
import pandas as pd
import glob
import difflib as df
import os
from pathlib import Path

### local variables ###
# if both false, still build the look-up tables needed!
# if true, run conversion (time: 5-10 mins), otherwise only build lookup table
conversion_flag = False

# if true save converted files to csv
csv_flag = False


# create directory for processed files
directory = "Clean2.0"
if not os.path.exists(directory):
    os.mkdir(directory)

### helper functions ###
def make_dict(lst1, lst2):
    return {lst1[i]: lst2[i] for i in range(len(lst1))}

def similar(a, b):
    return df.SequenceMatcher(None, a, b).ratio()

### class ###
class Data:
    # Replace raw FAOSTAT with GENuS
    # # FAOSTAT
    # # production and imports
    # prod = pd.DataFrame()
    # prod_imp = pd.DataFrame()
    # prod_US = pd.DataFrame()
    # prod_imp_US = pd.DataFrame()
    ### Original Data ###
    # GENuS
    GENuS = pd.DataFrame()

    # NHANES
    # another major db of interest
    demographics = pd.DataFrame()
    diets = pd.DataFrame()
    demoB = pd.DataFrame()
    dietB = pd.DataFrame()

    # FCID
    recipes = pd.DataFrame()
    food_des = pd.DataFrame()
    fcid_des = pd.DataFrame()
    codex = pd.DataFrame()

    # FlavorDB
    flavors = pd.DataFrame()
    flavors_des = pd.DataFrame()

    # Team created keys
    keys = pd.DataFrame()

    ###  Proccessed Dataframes ###
    #  225 food supply gram/person/day from 1961-2011
    """
    year, crop1, crop2, ...
    1961, x1.1, x1.2, ...
    1962, x2.1, x2.2, ...
    """
    GENuS_USA = pd.DataFrame()

    # 500+ FCID food commodities gram per dish ('food') eaten by a person ('SEQN') from 2001-2018
    """
    SEQN, food, var1, var2,..., ingredient 1, ingredient 2, …
    1, '75202022',x1, x2, x3, x4,  ...
    1, '54401100',y1, y2, y3, y4, ...
    """
    flat = pd.DataFrame()

    # 500+ FCID food commodities gram/person/day from 2001-2018
    """
    SEQN, ingredient 1, ingredient 2, …
    1, x1, x2, ...
    2, y1, y2, ...
    """
    flat_sum = pd.DataFrame()

    # UN-FAO 10 groups IDDS gram/person/day from 2001-2018
    # Note: can also treat the data as binary {x=0: False. x!=0, True}
    """
    SEQN, cat1, cat2, ...
    1, x1, y2
    2. y1, y2
    """
    flat_sum_10 = pd.DataFrame()

    # FlavorDB molecules gram/person/day from 2001-2008
    # Note: the calc of grams are summing any food that contains the compound,
    # therefore, it overestimates the actual amount. Try converting to binary as well
    """
    SEQN, compound 1, compound 2, …
    1, True, False, ...
    2, True, True, ...
    """
    flat_flavor = pd.DataFrame()

    #### Lookup Tables ###
    # these are accessible inside the class
    #faostatcode_to_faostatname = {}
    foodcode_to_foodname = {}
    fcidcode_to_fcidname = {}
    iddscode_to_iddsname = {}
    flavorDBcode_to_flavorDBname = {}
    moleculescode_to_moleculesname = {}

    def __init__(self):
        # Replace raw FAOSTAT with GENuS
        # self.prod = self.load_prod()
        # self.prod_imp = self.load_prod_imp()
        self.years, self.GENuS = self.load_multiple_csv("Data/GENuS/Edible_Food/")
        # data from B wave is loaded separately
        self.demographics = self.load_nhanes("Data/NHANES/Demographics/")
        self.diets = self.load_nhanes("Data/NHANES/WWEIA/")
        self.demoB = self.load_from_xpt("Data/NHANES/DEMO_B_0102.xpt")
        self.dietB = self.load_from_xpt("Data/NHANES/DRXIFF_B_0102.xpt")

        self.recipes = self.load_recipes()
        self.food_des = self.load_food_des()
        self.fcid_des = self.load_fcid_des()
        self.codex = self.load_codex()

        self.flavors = self.load_flavors()
        self.flavors_des = self.load_flavors_des()

        self.keys = self.load_from_csv("Data/FCID_to_All.csv")

    ### constructor body ###

    # # FAOSTAT (replaced by GENuS
    # def load_prod(self):
    #     df = self.load_from_csv("Data/FAOSTAT/production.csv")
    #     return df
    # def load_prod_imp(self):
    #     df = self.load_from_csv("Data/FAOSTAT/prodPlusImp.csv")
    #     return df

    # GENuS
    def load_multiple_csv(self, path):
        #all_files = glob.glob(path + "*.csv")
        all_files = Path(path).glob("*.csv")
        GENuS = pd.concat((pd.read_csv(f, na_values=['', '*']) for f in all_files), axis=0, ignore_index=True)
        #all_files = [str(f) for f in all_files]
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        years = [f.strip('.csv').split('_')[2] for f in csv_files]
        #print(years)
        return years, GENuS

    # NHANES
    def load_nhanes(self, path):
        #all_files = glob.glob(path + "*.xpt")
        all_files = Path(path).glob("*.xpt")
        list_of_pd = []
        for filename in all_files:
            df = self.load_from_xpt(filename)
            list_of_pd.append(df)
        #print(list_of_pd)
        nhanes = pd.concat(list_of_pd, axis=0, ignore_index=True)
        #print(nhanes)
        return nhanes

    # FCID
    def load_recipes(self):
        df = self.load_from_csv("Data/FCID/Recipes_WWEIA_FCID_0510.csv")
        return df
    def load_food_des(self):
        df = self.load_from_csv("Data/FCID/Food_Code_Description_0510.csv")
        return df
    def load_fcid_des(self):
        df = self.load_from_csv("Data/FCID/FCID_Code_Description_0510.csv")
        return df
    def load_codex(self):
        df = self.load_from_csv("Data/FCID/Codex_Lookup.csv")
        return df

    # FlavorDB
    def load_flavors(self):
        df = self.load_from_csv("Data/FlavorDB/flavordb.csv")
        return df
    def load_flavors_des(self):
        df = self.load_from_csv("Data/FlavorDB/molecules.csv")
        return df

    # load helpers
    def load_from_csv(self,filename):
        df = pd.read_csv(Path(filename), dtype=str)
        #print(df)
        return df
    def load_from_xpt(self,filename):
        df = pd.read_sas(filename)
        return df

data = Data()

#### data check ####
#print(data.demographics.head())
#print(data.demoB.keys())

#print(data.diets.head())
#print(data.dietB.keys())

# check if SEQN resets each wave
#data1_ids = data.diets['SEQN']
#data2_ids = data.dietB['SEQN']

# SEQN does not reset, but from histogram of SEQN, we found a gap!
#union = set(data1_ids).intersection(set(data2_ids))

### LOCAL CONVERSION TABLES ###
foodcode_to_fcidweights = {}
diets_to_weights = []
fcid_to_idds = {}
fcid_to_mddw = {}
fcid_to_flavorDB = {}
flavorDB_to_molecules = {}

# # 0. EXTRACT US ONLY DATA FROM FAOSTAT
# is_US_prod = data.prod['Area Code'] == '231' #'United States of America'
# is_US_prod_imp = data.prod_imp['Area Code'] == '231' #'United States of America'
#
# data.prod_US = data.prod[is_US_prod]
# data.prod_imp_US = data.prod_imp[is_US_prod_imp]
#
# if csv_flag:
#     data.prod_US.to_csv("Clean/data.prod_US.csv")
#     data.prod_imp_US.to_csv("Clean/data.prod_imp_US.csv")
#
# first_n_columns_prod  = data.prod_US.iloc[: , :6]
# first_n_columns_prod_imp  = data.prod_imp_US.iloc[: , :6]
#
# data.FAOSTAT_Description = first_n_columns_prod_imp.iloc[: , 3:]
#
# # Save the header to a sep. csv
# if csv_flag:
#     data.FAOSTAT_Description.to_csv('Clean/data.FAOSTAT_Description.csv', index=False)
#
# FAOSTAT_Description = data.FAOSTAT_Description.reset_index()
#
# # FAOSTAT Code to FAOSTAT Name
# codes = FAOSTAT_Description['Item Code']
# desc = FAOSTAT_Description['Item']
# for i in range(codes.shape[0]):
#      data.faostatcode_to_faostatname[codes[i]] = desc[i]#.strip().split(',')[0]

# 0. EXTRACT US-ONLY DATA FROM GENuS (GENuS replaces FAOSTAT)
if conversion_flag:
    is_USA = data.GENuS['Unnamed: 0'] == 'USA'
    data.GENuS_USA = data.GENuS[is_USA]
    data.GENuS_USA.insert(0,'Year',data.years)
    data.GENuS_USA = data.GENuS_USA.drop(['FOOD', 'Unnamed: 1'], axis=1, inplace=False)
    data.GENuS_USA = data.GENuS_USA.rename(columns = {'Unnamed: 0': 'Area'}, inplace=False)
    if csv_flag:
        data.GENuS_USA.to_csv(os.path.join(directory,'data.GENuS_USA.csv'))
        print('data.GENuS_USA.csv -- saved!')

# 1. CREATING LOOKUP TABLES
# Food code to food name
# print(data.food_des.keys())

codes = data.food_des['Food_Code']
desc = data.food_des['Food_Abbrev_Desc']

for i in range(codes.shape[0]):
    data.foodcode_to_foodname[codes[i]] = desc[i]#.strip('\'').split(',')

# FCID code to FCID name
codes = data.fcid_des['FCID_Code']
desc = data.fcid_des['FCID_Desc']

for i in range(codes.shape[0]):
    data.fcidcode_to_fcidname[codes[i]] = desc[i]#.strip('\'').split(',')

if conversion_flag:
    # Food Code to FCID weights (per 100 gram of food)
    foodcodes = list(set(data.recipes['Food_Code']))
    #len(foodcodes) # 7154 uniqe foods

    for code in foodcodes:
        foodcode_to_fcidweights[code] = {}

    for i in range(data.recipes.shape[0]):
        recipe = data.recipes.iloc[i]
        # print(i, int(recipe['FCID_Code']) ,recipe['Commodity_Weight'])
        try:
            foodcode_to_fcidweights[recipe['Food_Code']][recipe['FCID_Code']] = float(recipe['Commodity_Weight'])
        except:
            print(recipe['Food_Code'], data.foodcode_to_foodname[recipe['Food_Code']], 'missing')

    # 2. CONVERT FOOD FROM 24HR DIETARY RECALL TO FCID WEIGHTS
    # diets_to_weights = []

    # 2003-2018
    missing = 0
    print("START converting food into FCID weights 2003-2018...")
    for i in range(data.diets.shape[0]):
        # diets key description: https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DR1IFF_F.htm#DR1IFDCD
        row ={}
        diets = data.diets.iloc[i]
        # variables in row
        row['SEQN'] = str(int(diets['SEQN']))
        row['days'] = str(int(diets['DR1DAY'])) # day of the week
        row['home?'] = diets['DR1_040Z'].astype("int").astype("str") # 1: yes, 2:no, 7:refused, 9:dont know
        try:
            weights = foodcode_to_fcidweights[str(int(diets['DR1IFDCD']))].copy()
        except:
            #print(diets['DR1IFDCD'], ': missing')
            missing += 1

        # scale the weight by (eaten g / 100 g)
        scalar = diets['DR1IGRMS']/100
        #print(foodcode_to_foodname[diets['DR1IFDCD']],scalar)
        scaled_weights = ((x, y*scalar) for x, y in weights.items())
        row.update(scaled_weights)
        diets_to_weights.append(row)

    print("FINISH converting food into FCID weights 2003-2018...")

    print("START converting food into FCID weights 2001-2002...")
    # 2001-2002 (Note: variable names are different)
    for i in range(data.dietB.shape[0]):
        # diets key description: https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DR1IFF_F.htm#DR1IFDCD
        row ={}
        diets = data.dietB.iloc[i]
        # variables in row
        row['SEQN'] = str(int(diets['SEQN']))
        row['days'] = str(int(diets['DRDDAY'])) # day of the week
        row['home?'] = diets['DRD040Z'].astype("int").astype("str") # 1: yes, 2:no
        try:
            weights = foodcode_to_fcidweights[str(int(diets['DRDIFDCD']))].copy()
        except:
            #print(diets['DR1IFDCD'], ': missing')
            missing += 1

        # scale the weight by (eaten g / 100 g)
        scalar = diets['DRXIGRMS']/100
        #print(foodcode_to_foodname[diets['DR1IFDCD']],scalar)
        scaled_weights = ((x, y*scalar) for x, y in weights.items())
        row.update(scaled_weights)
        diets_to_weights.append(row)
    print("FINISH converting food into FCID weights 2001-2002...")

    # 2.1. CREATE A PANDA DATAFRAME FROM  DIETS_TO_WEIGHTS MULTI-LEVEL DICTIONARY
    print("START flattening multi-level dict into pd dataframe...")
    data.flat = pd.json_normalize(diets_to_weights)
    data.flat = data.flat.loc[:, data.flat.columns.notnull()]

    if csv_flag:
        data.flat.to_csv(os.path.join(directory,"data.flat.csv"))
        print('data.flat.csv -- saved!')

    print("FINISH flattening multi-level dict into pd dataframe...")

    # sum the rows by 'SEQN'
    print("START summing weights of FCID by SEQN...")
    data.flat_sum = data.flat.groupby('SEQN').sum()

    # drop ONE nan col
    data.flat_sum = data.flat_sum.loc[:, data.flat_sum.columns.notnull()]
    if csv_flag:
        data.flat_sum.to_csv(os.path.join(directory,"data.flat_sum.csv"))
        print('data.flat_sum.csv -- saved!')


    print("FINISH summing weights of FCID by SEQN...")

### MATCHING ATTEMPTS (not needed for final analysis) ###

# # 1. MATCH FAOSTAT <-> FCID
# # Too many edge cases for automatic matching. Using the I also realized that we do not need this yet.

# # 1.1 Fully automatic using squence matching
# #store the matches as a dict of FAOSTAT Code to FCID code
# faostatcode_fcidcode = {}
# for code in faostatcode_to_faostatname:
#     faostatcode_fcidcode[code] = None
#
# # adjust x for the matching percision
# x = 0.7
# for i in faostatcode_to_faostatname:
#     for j in fcidcode_to_fcidname:
#         if data.similar(faostatcode_to_faostatname[i].strip().split(',')[0],
#                    fcidcode_to_fcidname[j].strip().split(',')[0]) > x:
#             faostatcode_fcidcode[i] = j


# # 1.2 Semi-auto using two conversion tables
# # FCID -> Codex; Codex -> FAOSTAT

# # FCID -> Codex exists already
# #"Letter": "VR", "C_COMMONUM": "574", "Codex Code":"VR 0574"
# data.codex["Codex Code"] = data.codex["LETTER"].astype(str) + " " + data.codex["C_COMMONUM"].str.pad(4,fillchar='0')
#
# # Codex -> FAOSTAT, I found this online
# # http://www.who.int/foodsafety/areas_work/chemical-risks/IEDIcalculation0217clustersfinal.xlsm
# codexcode_to_faostatcode= {}
#
# # read data
# IEDI = pd.read_csv('Data/codex_to_faostat_from_IEDI.csv', dtype=str)
# codexcode = IEDI['Codex Code']
# faostatcode = IEDI['FAOstat FCLCode']
#
# for i in range(codes.shape[0]):
#      codexcode_to_faostatcode[codexcode[i]] = faostatcode[i]#.strip().split(',')[0]
#
# # delete empty rows
# codexcode_to_faostatcode = {k:v for k,v in codexcode_to_faostatcode.items() if v != '-'}
#
# # csv has a typo
# codexcode_to_faostatcode.pop('VD 0071 ')
# codexcode_to_faostatcode['VD 0071']='176'
#
# len(codexcode_to_faostatcode) # only 66 matches out of 500+
# # using the above dict to map the codex code
# data.codex["FAOSTAT Code"] = data.codex["Codex Code"].map(codexcode_to_faostatcode).fillna('')

# # Codex -> Faostat
# fcidcode_to_faostatcode= {}

# codes1 = data.codex['WWEIA_FCID_Code_0510']
# codes2 = data.codex['FAOSTAT Code']
# for i in range(codes.shape[0]):
#      fcidcode_to_faostatcode[codes1[i]] = codes2[i]#.strip().split(',')[0]
# data.fcid_des['FAOSTAT_Code'] = data.fcid_des["FCID_Code"].map(fcidcode_to_faostatcode)

## 3. MATCH FCID TO UNFAO-IDDS
# The automatic process could not account the complexities of food names and regional food differences.
# We follow the protocol here to manually convert 500+ FCID to UNFAO food groups,
# IDDS https://www.fao.org/3/i1983e/i1983e.pdf
# MDD-W https://www.fao.org/3/i5486e/i5486e.pdf
"""
    MATCH FCID Code to IDDS (9 groups) and MDD-W (10 groups)
    1. 1. A. cereal grains,
          B. white roots, tubers, and plantains
    8. 2. C. Pulses (beans, peas and lentils)
    8. 3. D. Nuts and Seeds
    9. 4. E. Milk and milk products
    6. 5. F. Organ Meat
    4.    G. Meat and poultry
          H. Fish and seafood
    7. 6. I. Eggs
    5. 7. J. Dark green leafy vegetables
    2. 8. K. Vitamin A-rich vegetables, roots and tubers
          L. Vitamin A-rich fruits
    3. 9. M. Other Vegetables
    3. 10 N. Other fruits
"""
data.iddscode_to_iddsname = {
    1:"Starchy staples",
    5:"Dark green leafy vegetables",
    2:"Other vitamin A rich fruits and vegetables",
    3:"Other fruits and vegetables",
    6:"Organ meat",
    4:"Meat and fish",
    7:"Eggs",
    8:"Legumes, nuts and seeds",
    9:"Milk and milk products"
}

if conversion_flag:
    # fcid_to_idds = {}
    codes1 = data.keys['FCID_Code']
    codes2 = data.keys['IDDS']
    for i in range(codes1.shape[0]):
         fcid_to_idds[codes1[i]] = codes2[i]#.strip().split(',')[0]

    # fcid_to_mddw = {}
    codes1 = data.keys['FCID_Code']
    codes2 = data.keys['MDD-W']
    for i in range(codes1.shape[0]):
         fcid_to_mddw[codes1[i]] = codes2[i]#.strip().split(',')[0]

    # rename the col name and sum the cols with the same name
    data.flat_10 = data.flat_sum.rename(columns=fcid_to_idds)
    data.flat_10= data.flat_10.groupby(lambda x:x, axis=1).sum()

    if csv_flag:
        data.flat_10.to_csv(os.path.join(directory,"data.flat_10.csv"))
        print('data.flat_10.csv -- saved!')

        # Save NHANES Demographics data
        data.demographics.to_csv(os.path.join(directory,"data.demographics.csv"))
        print('data.demographics.csv -- saved!')


# 4. MATCH FCID TO FlavorDB
# Same with 2. We manually match this and verify here.

# flavorDBcode_to_flavorDBname = {}
data.flavorDBcode_to_flavorDBname = make_dict(data.flavors['entity id'],data.flavors['alias'])

# moleculescode_to_moleculesname = {}
data.moleculescode_to_moleculesname = make_dict(data.flavors_des['pubchem id'],data.flavors_des['common name'] )

if conversion_flag:
    # fcid_to_flavorDB = {}
    codes1 = data.keys['FCID_Code']
    codes2 = data.keys['FlavorDB']
    for i in range(codes1.shape[0]):
         fcid_to_flavorDB[codes1[i]] = codes2[i]#.strip().split(',')[0]

    # flavorDB_to_molecules = {}
    codes1 = data.flavors['entity id']
    codes2 = data.flavors['molecules']

    # turn '{330, 403}' -> ['330', '403']
    for i in range(codes1.shape[0]):
         flavorDB_to_molecules[codes1[i]] = list(codes2[i][1:-1].split(", "))

    # moleculescode_to_moleculesname = {}
    data.moleculescode_to_moleculesname = make_dict(data.flavors_des['pubchem id'],data.flavors_des['common name'] )

    print("START Converting food to flavor...")
    favorDB_weight = data.flat_sum.rename(columns=fcid_to_flavorDB)
    favorDB_weight = favorDB_weight.groupby(lambda x:x, axis=1).sum()
    favorDB_1s0s = favorDB_weight.astype(bool).astype(int)

    for col in favorDB_1s0s:
        #print(col)
        for mol in flavorDB_to_molecules[str(col)]:
            if mol in data.flat_flavor:
                # update zero values only
                # data.flat_flavor[mol] = np.where(data.flat_flavor[mol].eq(0.0), favorDB_weight[col], data.flat_flavor[mol])
                # sum all values
                data.flat_flavor[mol] += favorDB_1s0s[col]
            else:
                data.flat_flavor[mol] = favorDB_1s0s[col]

    if csv_flag:
        data.flat_flavor.to_csv(os.path.join(directory,"data.flat_flavor.csv"))
        print('data.flat_flavor.csv-- saved!')

    print("End Converting food to flavor...")

## FINAL STATUS PRINTING
print("\nData Cleaning Completed!")
print("Are files converted?: ", conversion_flag)
print("Are files saved?: ", csv_flag)
if conversion_flag:
    print("\nConversion log:")
    print(str(len(data.diets) + len(data.dietB)) + ' records of food log')
    print(str(missing) + ' records of food log missing due to no standard recipe!')
    print(str(len(diets_to_weights)) + ' records of food log converted')
    print(str(len(data.flat_sum)) + ' out of ' + str(len(data.demographics)) + ' surveyed is in our analysis.')