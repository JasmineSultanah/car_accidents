import pandas as pd
import numpy as np
import logging

# !pip install geopandas
import geopandas as gpd
from shapely.geometry import Point

def save_obj(obj, name ):
    import pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def generate_substrings(data,col,n):
    '''
    This function generates new columns that are substrings of col. It will create n new columns. if n=3, the first column with suffix str1, will be a column with just the first character. Second will have two characters, and so on
    
    Parameters:
    data: pd.DataFrame. Data frame to manipulate
    col: col to create substrings of
    n: number of new cols to create
    '''
    new_cols = []
    for i in range(n):
        new_col_name = col+'_str'+str(i+1)
        data[new_col_name] = data[col].astype(str).str[:i+1]
        new_cols.append(new_col_name)       
            
    return new_cols

def generate_binned_features(data,rate_dict,rate_type='high'):
    '''
    This function creates new features in data based on the dictionary rate_dict. If the column for the key in rate dict contains variables in the value (list) for the key then it will be classed as 1 in the new column
    the new column will be called the same as the key but have a suffix of _high or _low based on rate type (whether the list means they have high or low rates of the target)
    
    Parameters:
    data: pandas.DataFrame to manipulate
    rate_dict: dict. features, with values that are the list of values that will be binned to one group
    rate_type: whether the values in the key have a high or low target rate. this will be the suffix for the new column
    '''
    
    
    new_cols = []
    for col, values in rate_dict.items():
        new_col = col+'_'+rate_type
        data[new_col] = np.where(data[col].isin(values),1,0)
        new_cols.append(new_col)
    return new_cols


def join_geojson(data,geojson_path):
    '''
    This function reads in the geojson from the geojson_path, creates a geometry column from the longitude and latitude columns of data and joins data to the geojson, where each row is situated inside the polygon (region) of a row in the geo_json
    
    Parameters:
    data: dataframe with columns for longitude and latitude
    geojson_path: path to geojson file
    
    Returns:
    geo_json: GeoDataFrame. What was read from the path
    data_geo: data with join to geojson, i.e. points mapped to a region in geojson
    
    '''
    # geo_json = gpd.read_file('https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/eurostat/ew/nuts2.json')
    geo_json = gpd.read_file(geojson_path)
    
    data = data.copy()
    # create a point geometry column using Point from the shapely package
    data['geometry'] = data.apply(
    lambda x: Point((x.longitude, x.latitude)),
    axis = 1)

    # set the crs - degrees
    data_crs = {'init': 'epsg:4326'}
    data_geo = gpd.GeoDataFrame(data,
    crs = data_crs,
    geometry = data.geometry)
    # convert the crs to match the geo_json (for some reason it doesn't work when one initially makes the geojson)
    data_geo = data_geo.to_crs(geo_json.crs)
    
    data_geo = gpd.sjoin(geo_json,data_geo,op='contains')
    
    return geo_json, data_geo


def data_preparation(file_path):
    logging.info('Loading in data')
    # read csv parsing the Date column as dates
    df = pd.read_csv(file_path,parse_dates=['Date'])
    
    df.rename(columns={'Weather_Conditions':'Weather','Urban_or_Rural_Area':'Urban_Rural','Did_Police_Officer_Attend_Scene_of_Accident':'Police_Officer_Attend','Road_Surface_Conditions':'Road_Surface'},inplace=True)

    all_dfs_dict = pd.read_excel('variable lookup.xls',sheet_name=None)
    
    for col in df.columns:
    
        cleaned_keys = [ i.replace(' ','_') for i in all_dfs_dict.keys()]

        if col in cleaned_keys:
            print(col)
            old_col = col.replace('_',' ')
            lookup_df = all_dfs_dict[old_col].copy()
            lookup_df.columns = [i.lower() for i in lookup_df.columns]
            lookup_df['code'] = lookup_df['code'].astype(object)

            df[col] = df[col].astype(object)

            print(df.shape)
            lookup_df.rename(columns={'code':col,'label':col+'_Name'},inplace=True)

            df = df.merge(lookup_df,how='left').copy()
            df.drop(col,inplace=True,axis=1)

            df.rename(columns={col+'_Name':col},inplace=True)

        
    
    # turn all column names to lower case
    df.columns = [i.lower() for i in df.columns]

    # rename did_police_officer_attend_scene_of_accident to 'target', so it's less typing!
    if 'police_officer_attend' in df.columns:
        df.rename(columns={'police_officer_attend':'target'},inplace=True)
        df['target'] = np.where(df['target']=='Yes',1,0)

    # generate time based variables
    # Time
    df['hour'] = df['time'].str[:2]
    df['minutes_past_hour'] = df['time'].str[-2:]
    df['month'] = df.date.dt.month
    df['15_mins_past_hour'] = np.where(df['minutes_past_hour'].isin(['00','15','30','45']),1,0)
    df['weekday']= np.where(((df['day_of_week']!='Saturday' ) &(df['day_of_week']!='Sunday')),1,0)
    df['night_time'] = np.where(df['hour'].isin(['23','00', '01', '02', '03', '04', '05', '06']),1,0)
    
    # generate binned variables for numeric columns
    df['number_of_casualties_more_than_2'] = np.where(df['number_of_casualties'].astype(int)>2,1,0)
    df['number_of_vehicles_more_than_2'] = np.where(df['number_of_vehicles'].astype(int)>2,1,0)
    
    location_cols = []

    substring_col_mapping = {'location_easting_osgr':1, 'location_northing_osgr':1}


    for key, value in substring_col_mapping.items():
        location_cols = location_cols + generate_substrings(df,key,value)
    
    high_rate_dict = load_obj('high_rate_dict')
    low_rate_dict = load_obj('low_rate_dict')
    
    binned_high_cols = generate_binned_features(df,high_rate_dict,rate_type='high')
    binned_low_cols = generate_binned_features(df,low_rate_dict,rate_type='low')
    
    geo_json, geo_data = join_geojson(df,geojson_path='https://datahub.io/core/geo-nuts-administrative-boundaries/r/nuts_rg_60m_2013_lvl_3.geojson')
    
    df = df.merge(geo_data[['accident_index','NUTS_ID']],how='left')
    
    df['NUTS_ID_high'] = np.where(df['NUTS_ID'].isin(['UKD46', 'UKD41', 'UKM64', 'UKD47', 'UKD45', 'UKD44', 'UKD42', 'UKL15',
            'UKM71', 'UKL18', 'UKL22', 'UKL17', 'UKM61', 'UKM91', 'UKM77', 'UKM63',
            'UKM92', 'UKL24']),1,0)
    df['NUTS_ID_low'] = np.where(df['NUTS_ID'].isin(['UKJ31', 'UKG37', 'UKD61', 'UKG31', 'UKI53', 'UKJ28', 'UKG21', 'UKG33',
            'UKJ44', 'UKC13', 'UKF21', 'UKG32', 'UKC22', 'UKJ27', 'UKF11', 'UKF14',
            'UKK11', 'UKK21', 'UKE11', 'UKJ21']),1,0)
    
    return df
    