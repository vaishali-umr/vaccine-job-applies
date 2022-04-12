import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from bokeh.plotting import figure, show
from bokeh.palettes import Spectral5

from ipywidgets import interact, interact_manual
from ipywidgets import RadioButtons

from bokeh.io import output_notebook
output_notebook()

path = r'C:\Users\16302\Documents\Coding Samples\vaccine-job-applies'

def load_job_activity(path, fname):
    job_activity = pd.read_csv(os.path.join(path, fname))
    job_activity = job_activity.drop(['Date', 'Enterprise ID', 'Currency', 'Employer ID'], axis=1)  
    job_activity = job_activity.loc[(job_activity['Body ID'] != '00000000-0000-0000-0000-000000000000')]
    return job_activity

def load_vaccine_ref(path, fname):
    vaccine_ref = pd.read_csv(os.path.join(path, fname))
    vaccine_ref = vaccine_ref.loc[(vaccine_ref['Tag'] != '2')]
    vaccine_ref = vaccine_ref.dropna(subset=['Tag'])
    return vaccine_ref

def concatenate(fnames):
    long = pd.concat(fnames, ignore_index=True)
    return long

def merge(job_activity, vaccine_ref):
    df = job_activity.merge(vaccine_ref, on='Body ID', how='outer')
    df = df.drop(['Matches'], axis=1)
    df['Tag'] = df['Tag'].fillna('Not Applicable') #change to 'Not Applicable' or 'No Vaccine Mentioned' depending on separation
    df.to_csv('merged_df.csv.zip', index=False, compression='zip')
    
    return df

def calc_cta(df):
    df['Click to Apply (%)'] = (df['Paid Applies']/df['Paid Clicks'])*100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['Click to Apply (%)'])
    return df
    
def diff_cta(df, on):
    df_na = df.query('Tag=="Not Applicable"').rename(columns={'Click to Apply (%)':'CTA NA'})
    df_covid = df.query('Tag=="COVID Vaccine Required"').rename(columns={'Click to Apply (%)':'CTA COVID'})
    df_diff = df_na[[on, 'CTA NA']].merge(df_covid[[on, 'CTA COVID']], on=on, how='inner')
    df_diff['Difference in CTA (%)'] = df_diff['CTA NA']-df_diff['CTA COVID']
    return df_diff

def aggregate_cta(df):
    agg_cta = df.groupby(['Tag'])['Paid Clicks', 'Paid Applies', 'Count of Jobs'].apply(lambda x : x.sum()).reset_index()
    agg_cta = calc_cta(agg_cta)
    return agg_cta

def by_category(df, category):
    result = df.groupby([category, 'Tag'])['Paid Clicks', 'Paid Applies', 'Count of Jobs'].apply(lambda x : x.sum()).reset_index()
    result = calc_cta(result)
    return result

def multiple_subplot(df, x, y1, y2, y3, y4):
    fig, ax = plt.subplots(2, 2, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    
    ax[0,0].bar(df[x], df[y1])
    ax[0,0].set(ylabel=y1)
    
    ax[0,1].bar(df[x], df[y2])
    ax[0,1].set(ylabel=y2)
    
    ax[1,0].bar(df[x], df[y3])
    ax[1,0].set(ylabel=y3)
    
    ax[1,1].bar(df[x], df[y4])
    ax[1,1].set(ylabel=y4)
    
    fig.suptitle('Various Job Activity by Vaccine Mention')
    
    for ax in ax.flat:
        ax.set_xticklabels(df['Tag'], rotation=90)
        
    fig.savefig('multiple_subplot.png', bbox_inches='tight')

def bar_plot(df, x, y, x_axis):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_style('darkgrid')
    
    graph = sns.barplot(x, y, data=df, palette ='tab10', ax=ax)
    graph.set_title('Change in Click to Apply (%) When COVID Vaccine Not Required', fontsize=20)
    graph.set_xlabel(x_axis, fontsize=18)
    graph.set_ylabel('Difference in Click to Apply (%)', fontsize=18)
    graph.set_xticklabels(graph.get_xticklabels(), rotation=90, horizontalalignment='right', size=15)
    graph.set_yticklabels(graph.get_yticks(), size=15)

    fig.savefig(f'diff_cta_{x_axis}.png', bbox_inches='tight')

#@interact(Industry=industries, Metric=RadioButtons(options=['Clicks', 'Applies', 'Click to Apply (%)']))
def interact_plot_industry(Industry='Administration', Metric='Clicks'):
          
    if Metric == 'Clicks':
        col='Paid Clicks'
    elif Metric == 'Applies':
        col='Paid Applies'
    elif Metric == 'Click to Apply (%)':
        col='Click to Apply (%)'
    
    subset = df_industry.loc[df_industry['Discipline Name'] == Industry]
    tags = [each for each in subset['Tag'].unique()]
    values = [each for each in subset[col]]
    
    plot = figure(x_range=tags, title=f'{col} in {Industry}', y_axis_label=Metric)
    plot.vbar(x=tags, top=values, width=0.8, color=Spectral5)
    show(plot)
    
#@interact(Category=interested_cat, Metric=RadioButtons(options=['Clicks', 'Applies', 'Click to Apply (%)']))
def interact_plot_job_function(Category='29-1141.00 - Registered Nurses', Metric='Clicks'):
          
    if Metric == 'Clicks':
        col='Paid Clicks'
    elif Metric == 'Applies':
        col='Paid Applies'
    elif Metric == 'Click to Apply (%)':
        col='Click to Apply (%)'
    
    subset = df_job_cat.loc[df_job_cat['Tagged Category'] == Category]
    tags = [each for each in subset['Tag'].unique()]
    values = [each for each in subset[col]]
    
    plot = figure(x_range=tags, title=f'{col} for {Category}', y_axis_label=Metric)
    plot.vbar(x=tags, top=values, width=0.8, color=Spectral5)
    show(plot)

def get_dummy(df):
    with_dummy = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Tag'], drop_first=True)    
    return with_dummy

def ols(df, x, y):   
    x = df[x]
    y = df[y]
    
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    print(model.summary())

### end functions

#load
sept_job_activity = load_job_activity(path, 'September Job Activity Data-edited.csv')
sept_vaccine_ref = load_vaccine_ref(path, 'September Vaccine Reference Data.csv')
oct_job_activity = load_job_activity(path, 'October Job Activity Data-edited.csv')
oct_vaccine_ref = load_vaccine_ref(path, 'October Vaccine Reference Data.csv')

#concatenate, merge, and output to csv
job_activity = concatenate([sept_job_activity, oct_job_activity])
job_activity = job_activity.drop(['Discipline ID'], axis=1)  
vaccine_ref = concatenate([sept_vaccine_ref, oct_vaccine_ref])

df = merge(job_activity, vaccine_ref)

#aggregate data
agg_cta = aggregate_cta(df)

#industry level
df_industry = by_category(df, 'Discipline Name')

#create list of unique industries
industries = [each for each in df_industry['Discipline Name'].unique()]

#job function level
df_job_cat = by_category(df, 'Tagged Category')

#create list of job categories of interest
my_cat = ['29-1141.00 - Registered Nurses',
                 '35-3031.00 - Waiters and Waitresses',
                 '41-2031.00 - Retail Salespersons',
                 '43-4051.00 - Customer Service Representatives',
                 '49-3023.00 - Automotive Service Technicians and Mechanics',
                 '49-9071.00 - Maintenance and Repair Workers, General',
                 '53-3031.00 - Driver/Sales Workers',
                 '53-3032.00 - Heavy and Tractor-Trailer Truck Drivers',
                 '53-3033.00 - Light Truck or Delivery Services Drivers']

#subset based on those categories and clean titles
df_my_cat = df_job_cat[df_job_cat['Tagged Category'].isin(my_cat)]
df_my_cat['Tagged Category'] = df_my_cat['Tagged Category'].map(lambda x: x[13:])

#create list of unique job categories
categories = [each for each in df_my_cat['Tagged Category'].unique()]

#find difference in cta's
df_diff_industry = diff_cta(df_industry, 'Discipline Name')
df_diff_my_cat = diff_cta(df_my_cat, 'Tagged Category')

#multiple subplot
multiple_subplot(agg_cta, 'Tag', 'Count of Jobs', 'Paid Clicks', 'Paid Applies', 'Click to Apply (%)')

#difference in cta barplots
bar_plot(df_diff_industry, 'Discipline Name', 'Difference in CTA (%)', 'Industry')
bar_plot(df_diff_my_cat, 'Tagged Category', 'Difference in CTA (%)', 'Job Category')

#see jupyter file for interactive plots!

#create dummy variable and ols
with_dummy = get_dummy(df_industry)

ols(with_dummy, 'Not Applicable', 'Click to Apply (%)')
ols(with_dummy, ['Not Applicable', 'Count of Jobs'], 'Click to Apply (%)')
