#%% packages
from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
from plotnine import ggplot, aes, geom_line, labs, theme, element_text, geom_map, theme_minimal,scale_fill_gradientn, ylim
import pandas as pd

#%%
world_wind = pd.read_pickle('world_wind.pkl')
world_wind
# values for input fields
year_min = 2014
year_max = world_wind['Year'].max()
#%% values for dropdown field
country_names = world_wind['Country'].unique()
country_names = country_names[1: ]  # leave out first element
country_names_dict = {l:l for l in country_names}


# %%
app_ui = ui.page_fluid(
    ui.panel_title('Wind Energy Installations'),
    ui.p('This dashboard shows wind energy installations over time, and by country.'),
    ui.span('Data is based on '),
    ui.a("Wikipedia", href="https://en.wikipedia.org/wiki/Wind_power_by_country", target="_blank"),
    ui.br(),
    ui.br(),
    ui.navset_pill(
        ui.nav("By Region", 
            ui.input_slider(id="year", label="year", max=year_max, min=year_min, value=year_max, step=1),
            ui.output_plot('plot_map')),
        ui.nav("Over Time", 
          ui.input_selectize(id='country', label='Country', choices=country_names_dict, selected='China', multiple=True),
          ui.output_plot('plot_installations_time')
          
    )
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def wind_country():
        df_filt = world_wind.loc[((world_wind['Country'].isin(input.country()))),:]
        return df_filt

    @reactive.Calc
    def wind_time():
        year_selected = input.year()
        df_filt = world_wind.loc[(world_wind['Year']==year_selected),:]
        return df_filt


    @output
    @render.plot
    def plot_installations_time():
        g =ggplot(wind_country(), aes('Year', 'Installation', color='Country')) + geom_line() + theme(axis_text_x=element_text(rotation=90, hjust=1)) + labs(x = 'Year', y='Installations [MW]', title='Wind Installations by Country') + theme_minimal() + ylim([0, 330000])
        return g
    
    @output
    @render.plot
    def plot_map():
        g = ggplot(data=wind_time(), mapping=aes(fill="Installation")) + geom_map() + theme_minimal() + scale_fill_gradientn(limits=[0, 200000], colors=["#2cba00", "#a3ff00", "#fff400", "#fff400", "#ff0000"])
        return g


app = App(app_ui, server)
