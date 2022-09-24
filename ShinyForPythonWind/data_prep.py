#%% packages
import pandas as pd
import geopandas as gpd
import numpy as np
from plotnine import ggplot, aes, geom_map, theme_minimal, scale_fill_gradientn

#%%
wind = pd.read_csv('Wind.csv')
wind.loc[(wind['Country']== 'United States'), 'Country']= 'United States of America'
wind['Country'] = wind['Country']
wind.loc[(wind['Country']=='Norway[35][36]'), 'Country'] = 'Norway'
#%% adapt columns
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world_wind = world.merge(wind, left_on='name', right_on='Country', how='left')
world_wind = world_wind.replace(np.nan, 0)


# %% save data
world_wind.to_pickle('world_wind.pkl')
# %% test data import
world_wind = pd.read_pickle('world_wind.pkl')
world_wind
# %% Test
df = world_wind.loc[(world_wind['Year']==2021),:]
g =ggplot(data=df, mapping=aes(fill="Installation")) + geom_map() + theme_minimal() + scale_fill_gradientn(limits=[0, 200000], colors=["#2cba00", "#a3ff00", "#fff400", "#fff400", "#ff0000"])
g

# %%
from plotnine.themes.theme_minimal