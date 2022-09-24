# Download Wind Installation table
url = 'https://en.wikipedia.org/wiki/Wind_power_by_country'
xpath = '//*[@id="mw-content-text"]/div[1]/table[2]'

library("rvest")
library(dplyr)
library(stringr)
library(tidyr)
library(readr)

wind <- read_html(url) %>% 
  html_node(xpath =  xpath) %>% 
  html_table(fill = T)

wind_mod <- wind %>% 
  select(-`#`) %>%
  mutate_at(vars(`Country or territory`), ~ str_replace(., "\\[[0-9]+\\]", "")) %>% 
  mutate(`Country or territory` = as.factor(`Country or territory`)) %>% 
  mutate_at(vars(`2014[25]`, `2015[3]`, `2016[26]`, `2017[27]`, `2018[28]`, `2019[16]`, `2020[29]`, `2021[30]`), ~ str_replace(., ",", "")) %>% 
  mutate_at(vars(`2014[25]`, `2015[3]`, `2016[26]`, `2017[27]`, `2018[28]`, `2019[16]`, `2020[29]`, `2021[30]`), ~ str_replace(., "\\*", "")) %>% 
  mutate_if(is.character,as.numeric) %>% 
  filter(`Country or territory` != 'World total capacity (MW)') %>% 
  drop_na() %>% 
  pivot_longer(cols = 2:ncol(.), names_to = 'Year', values_to = 'Installation') %>% 
  mutate(Year = str_replace(Year, "\\[[0-9]+\\]", "")) %>% 
  mutate(Year = as.numeric(Year))
wind_mod

write.csv(wind_mod, file = 'Wind.csv', row.names = F)
