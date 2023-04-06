import pandas as pd
import geopandas as gpd

import requests
import json

import re
from pathlib import Path

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-GPC': '1',
}

def getMunicipalities():
       
    with requests.Session() as s:
        rs = s.get('https://ws.geonorge.no/kommuneinfo/v1/kommuner', headers=headers)
        parse = json.loads(rs.text)
    
    df = pd.DataFrame.from_records(parse)
    return df

def searchMunicipality(query):

    params = (
        ('knavn', query),
    )

    with requests.Session() as s:
        rs = s.get('https://ws.geonorge.no/kommuneinfo/v1/sok', headers=headers, params=params)
        parse = json.loads(rs.text)

    return parse

def municipalityInfo(munId):

    with requests.Session() as s:
        rs = s.get(f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{munId}', headers=headers)
        parse = json.loads(rs.text)

    return parse

def findMunicipality(pnt):
    
    params = (
        ('ost', pnt.x),
        ('koordsys', '4258'),
        ('nord', pnt.y),
    )
    
    with requests.Session() as s:
        rs = s.get('https://ws.geonorge.no/kommuneinfo/v1/punkt', headers=headers, params=params)
        parse = json.loads(rs.text)
    
    return parse


def municipalityPolys(munList):
    
    dfMun = getMunicipalities()
    pattern = r'(?<=\()[\w\s]+(?=\)$)'

    def getFeature(mun):
        
        temp = mun.split(' (')[0]
        
        mask = dfMun['kommunenavnNorsk'] == temp
        munId = dfMun.loc[mask]
        
        if len(munId) > 1:
            cnty = re.search(pattern, mun).group()
            
            hit = None
            for i in range(len(munId)):
                munInfo = municipalityInfo(munId.iloc[i]['kommunenummer'])
                if munInfo['fylkesnavn'] == cnty:
                    hit = i
                    break
                    
            munId = munId.iloc[i]['kommunenummer']
        
        else:
            munId = munId['kommunenummer'].squeeze()
        
        url = f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{munId}/omrade'
        with requests.Session() as s:
            rs = s.get(url, headers=headers)
            parse = json.loads(rs.text)

        feature = {'properties': {'munId': munId, 'municipality': mun}, 'geometry': parse['omrade']}
        return feature

    path = Path.cwd() / 'data' / 'dgi' / 'nor_municipalities.json'

    if not path.exists():

        features = []
        for mun in munList:
            features.append(getFeature(mun))
        
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf.set_crs(epsg=4258, inplace=True)
        #gdf['municipality'] = munList
        gdf.to_file(path, driver='GeoJSON', encoding='utf-8')

    else:
        gdf = gpd.read_file(path)

        newMun = set(munList).difference(set(gdf['municipality'].unique()))
        
        if newMun:
            features = []
            for mun in newMun:
                features.append(getFeature(mun))

            newGdf = gpd.GeoDataFrame.from_features(features)
            newGdf.set_crs(epsg=4258, inplace=True)
            newGdf['municipality'] = newMun

            gdf = gdf.append(newGdf, ignore_index=True)
            gdf.to_file(path, driver='GeoJSON', encoding='utf-8')

    return gdf