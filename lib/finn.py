HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
  'Accept': '*/*',
  'Accept-Language': 'en-US,en;q=0.5',
  'DNT': '1',
  'Connection': 'keep-alive',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-origin',
  'Sec-GPC': '1',
}

def finnData():
    
  def parseJson(docs):
      
    scrap = []
    for doc in docs:
      lat = doc['coordinates']['lat']
      lon = doc['coordinates']['lon']
      
      if int(lat) == 0 or int(lon) == 0:
        continue
      
      pnt = Point(lon, lat)
      
      area = doc.get('area_range', doc.get('area'))
      area = area.get('size_from', area.get('size'))
      
      if area == 0:
        continue
      
      price = {}
      for sfx in ['total', 'suggestion']:
        temp = doc.get(
          f'price_{sfx}', 
          doc.get(f'price_range_{sfx}')
        )
        price[f'price_{sfx}'] = temp.get('amount', temp.get('amount_from'))

      if 0. in price.values():
        continue
          
      shCst = doc.get('price_shared_cost', np.nan)
      if isinstance(shCst, dict):
        shCst = shCst.get('amount')
          
      beds = doc.get(
        'number_of_bedrooms', 
        doc.get('bedrooms_range')
      )
      if isinstance(beds, dict):
        beds = beds.get('start')
      
      # Municipality
      #mun = doc['location'].split(', ')[-1]
      #if mun not in setMun:
      #    
      #    if mun not in dctMun:
      #        temp = gn.findMunicipality(pnt)['kommunenummer']
      #        dctMun[mun] = temp
      #        mun = temp
      #    else:
      #        mun = dctMun[mun]
      
      scrap.append({
        'id': doc['ad_id'],
        'timePublished': doc['timestamp'],
        'geometry': pnt,
        'address': doc['location'],
        #'municipality': mun,
        'priceTotal': price['price_total'],
        'priceAsk': price['price_suggestion'],
        'sharedCost': shCst,
        'area': doc['area_range']['size_from'],
        'bedrooms': beds,
        'propertyType': doc['property_type_description'],
        'ownerType': doc['owner_type_description'],
        'link': doc['ad_link']
      })
    return scrap
  
  def iteratePages(scrap, params, startPage):
      
    for p in range(startPage,51):
      params[-1] = ('page', str(p))
  
      with requests.Session() as s:
        rs = s.get(
          'https://www.finn.no/api/search-qf', 
          headers=HEADERS, params=params
        )
        parse = json.loads(rs.text)
      
      if not 'docs' in parse['docs']:
        continue
      
      scrap.extend(parseJson(parse['docs']))
            
    if parse['docs']:
      last = parse['docs'][-1]
      priceTo = last.get('price_suggestion', last.get('price_range_suggestion'))
      priceTo = priceTo.get('amount', priceTo.get('amount_from'))
    
    else:
      priceTo = 0
    
    return scrap, priceTo
      
  params = {
    'searchkey': 'SEARCH_ID_REALESTATE_HOMES',
    'lifecycle': '1',
    'property_type': ['1', '2', '3', '4'],
    'sort': 'PRICE_ASKING_DESC',
    'price_to': '',
    'page': '1',
  }
  
  with requests.Session() as s:
      rs = s.get('https://www.finn.no/api/search-qf', headers=headers, params=params)
      parse = json.loads(rs.text)
  
  nUnits = parse['metadata']['result_size']['match_count']
  
  scrap = []
  scrap.extend(parseJson(parse['docs']))
  
  scrap, priceTo = iteratePages(scrap, params, 2)
      
  while (priceTo > 0) and (len(scrap) <= nUnits):
      params[-2] = ('price_to', str(priceTo))
      scrap, priceTo = iteratePages(scrap, params, 1)
  
  gdf = gpd.GeoDataFrame(scrap, crs=4258)
  gdf.drop_duplicates(inplace=True)
  gdf['priceArea'] = gdf['priceTotal'] / gdf['area']

  # Additional data
  for scope in {'municipality', 'postalarea'}:
      path = Path.cwd() / 'data' / 'dgi' / f'virdi_{scope}.json'
      parser = partial(virdiChoroPolys, scope)
      choroPolys = loadGeodata(parser, path, relativedelta(months=6))

      fld = 'postalCode' if scope == 'postalarea' else scope

      gdf = gdf.sjoin(choroPolys[['geometry', fld, f'price{scope.capitalize()}']], how='left', predicate='within')
      gdf.drop('index_right', axis=1, inplace=True)

      # Price delta
      gdf[f'delta{scope.capitalize()}'] = gdf['priceArea'] - gdf[f'price{scope.capitalize()}']
      gdf.drop(f'price{scope.capitalize()}', axis=1, inplace=True)

  return gdf
