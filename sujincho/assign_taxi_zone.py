import geopandas  
from shapely.geometry import Point
import numpy as np

def assign_taxi_zones(df,lon_var,lat_var):

    localdf=df[[lon_var,  lat_var]].copy()

    localdf[lon_var]=localdf[lon_var].fillna(value=0.)
    localdf[lat_var]=localdf[lat_var].fillna(value=0.)

    shape_df=geopandas.read_file('./NYC_Taxi_Zones/NY_taxi_zone.shp')#.shp 파일이지만 다운 받은 압축 푼 파일에 4가지 파일(.dbf, .prj, .shp, .shx)가 같은 폴더에 함께 있어야함 
    shape_df.drop(['objectid',"shape_area","shape_leng","borough","zone"],axis=1, inplace=True)
    shape_df=shape_df.to_crs({'init':'epsg:4326'})

    try:
        
        local_gdf=geopandas.GeoDataFrame(localdf,crs={'init':'epsg:4326'},geometry=[Point(xy) for xy in zip(localdf[lon_var],localdf[lat_var])])
        local_gdf=geopandas.sjoin(local_gdf,shape_df,how='left',op='within')

        if local_gdf.location_i.isnull:

            local_gdf.location_i=local_gdf.location_i.fillna(value=0)


        return local_gdf.location_i.astype(int)

    except ValueError as ve:
        print(ve)
        #print(ve.stacktrace())
        series=localdf[lon_var]
        series=np.nan
        return series


