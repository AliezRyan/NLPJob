# Public Transport: Bus and MTR Service Availability in Hong Kong

## Proposed Title

**Visual Analysis of Bus and MTR Service Availability Across Hong Kong's 18 Districts**

## Project Objective

This project investigates how evenly bus and MTR services are distributed across Hong Kong's 18 districts. Interactive visualizations will compare transport availability with population and socioeconomic characteristics, identify districts with relatively low provision, and test whether conclusions change across absolute, per-capita, and area-based measures of service availability.

The project focuses on buses and the MTR because they form the core of Hong Kong's land-based public transport network and provide sufficient geographic and service diversity for district-level visual analysis. The study will identify relative differences between districts; it will not claim to determine an official or absolute standard of transport adequacy.

## Research Questions and Hypotheses

The visual analysis will address the following questions:

1. How do the absolute numbers of bus stops, bus routes, and MTR stations differ across the 18 districts?
2. Which districts have relatively low transport provision after normalization by population or land area?
3. Does a district's ranking change when absolute, per-capita, and area-based measures are compared?
4. Are transport availability measures associated with population, median employment income, or employment indicators?
5. Do bus services appear to complement limited MTR coverage in some districts?
6. How does the availability of barrier-free MTR facilities vary across districts?

The project will explore three initial hypotheses:

- **H1:** More populous districts have more transport facilities in absolute terms but not necessarily more facilities per 100,000 residents.
- **H2:** District rankings differ substantially between population-normalized and land-area-normalized measures.
- **H3:** Districts with fewer MTR stations tend to depend more heavily on bus stops and routes.

These hypotheses are exploratory. Observed associations will not be interpreted as evidence that transport provision causes income or employment outcomes.

## Dataset Description

### 1. District Population and Socioeconomic Data

The 2021 Population Census provides district-level population, labour-force, working-population, age, household, and income indicators. It will supply denominators for per-capita measures and variables for socioeconomic comparisons.

The analysis will use **median monthly employment income**, which is the income measure available in the census data. An unemployed-population indicator may be derived as labour force minus working population and will be labelled as a derived measure.

The 2016 Population By-census may be used to show changes in demographic demand between 2016 and 2021. It will not be used to imply changes in transport supply unless comparable historical transport data are obtained.

### 2. District Boundary Data

Hong Kong's 18 District Council district polygons will provide a common geographic unit. Bus stops and MTR stations will be assigned to districts through point-in-polygon spatial joins. District land area will be calculated in a suitable projected coordinate system for area-based indicators.

### 3. Bus Data

The Transport Department's bus-route and bus-stop-location datasets provide route identifiers, operators, stop identifiers, fares, service information, point coordinates, and route geometry. They will support:

- mapping bus stops and selected route paths;
- counting unique bus stops in each district;
- counting unique bus routes that serve at least one stop in each district; and
- comparing bus provision using absolute, per-capita, and area-based measures.

The official bus datasets must be downloaded and checked before implementation. Route identifiers, directions, operators, duplicate stop records, coordinate order, and update dates will be validated before aggregation.

### 4. MTR Data

The MTR dataset provides line codes, station identifiers, station sequences, fares, and barrier-free facility information. The current station file does not contain coordinates or district identifiers. A validated station-coordinate dataset must therefore be obtained and joined by station ID or English station name before geographic MTR analysis begins.

Railway line geometry may be used as a contextual map layer, but station points rather than line intersections will be used to count MTR stations by district. Interchange stations will be counted once when measuring physical stations, even if they serve multiple lines.

### 5. Monthly Traffic and Transport Digest

The Monthly Traffic and Transport Digest may provide a separate, system-wide time-series view of bus and railway patronage. Since its main passenger statistics are reported by operator rather than district, it will not be used as evidence of district-level usage or adequacy.

## Metric Definitions

| Metric | Definition |
|---|---|
| Bus stops in a district | Number of unique bus-stop IDs whose point locations fall within the district |
| Bus routes serving a district | Number of unique route IDs with at least one stop in the district, after applying a consistent rule for direction and operator |
| MTR stations in a district | Number of unique physical station IDs whose point locations fall within the district |
| Stops or stations per 100,000 residents | Unique facility count divided by district population, multiplied by 100,000 |
| Routes per 100,000 residents | Unique routes serving the district divided by district population, multiplied by 100,000 |
| Facilities per square kilometre | Unique facility count divided by district land area in square kilometres |
| Bus-to-MTR provision ratio | District bus-stop availability divided by MTR-station availability, with districts containing no MTR station shown separately |
| Barrier-free facility availability | Counts or proportions of selected barrier-free facility categories at MTR stations in each district |

Per-capita results will be expressed as rates per 100,000 residents. Population-normalized and area-normalized measures will be shown together because Hong Kong districts differ greatly in population and physical size.

## Methodology

### 1. Data Preparation

Pandas will be used to clean tabular data, standardize identifiers and district names, remove aggregate rows, and derive indicators. GeoPandas and Shapely will be used to validate coordinates, transform coordinate reference systems, perform point-in-polygon spatial joins, and aggregate transport features by district.

The preparation process will:

1. retain the 18 district records and remove territory-wide aggregate records;
2. validate district polygons and coordinate-axis order;
3. deduplicate bus stops by stop ID and MTR stations by physical station ID;
4. define a consistent route-counting rule for operators and directions;
5. assign stops and stations to districts through spatial joins;
6. calculate population-normalized and land-area-normalized indicators; and
7. record the reference date of every dataset.

### 2. Geographic Visualizations

GeoPandas, Folium, and Plotly will be used to create:

- an interactive map with district boundaries and clustered bus stops;
- selectable bus-route layers using simplified or filtered route geometry;
- MTR station and railway-line layers after station coordinates are obtained;
- choropleth maps of stops, routes, and stations per 100,000 residents;
- area-normalized maps of facilities per square kilometre; and
- a bivariate or linked comparison showing bus and MTR provision together.

The raw bus-route geometry will be processed offline and will not be loaded directly into the browser. Route geometry will be simplified, filtered after user selection, or converted to an efficient display format.

### 3. Statistical and Exploratory Visualizations

Plotly, Matplotlib, and Seaborn will be used to create:

- ranked bar charts for absolute, per-capita, and area-based provision;
- slope or rank-change charts showing how district rankings change between metrics;
- scatter plots comparing transport indicators with population, median employment income, and employment indicators;
- coordinated bus-versus-MTR plots for investigating possible modal complementarity;
- charts comparing barrier-free MTR facility availability; and
- an optional system-wide patronage time series clearly separated from district analysis.

Interactive selections will link maps and charts. Selecting a district will reveal its raw counts, normalized indicators, demographic context, and ranking under different definitions. This supports hypothesis generation and verification by allowing users to compare alternative explanations instead of viewing isolated charts.

### 4. Analytical Approach

The analysis will begin with absolute counts and then test whether the interpretation changes after normalization. Pearson or Spearman correlations may be reported as descriptive summaries where appropriate, accompanied by scatter plots and clear warnings about small sample size and non-causal interpretation. With only 18 districts, visual inspection, outlier analysis, and sensitivity comparisons will be emphasized over complex predictive modelling.

## Feasibility and Scope Control

The bus-focused district analysis is achievable once the official route and stop datasets are downloaded. The census file and district boundaries are already available. Complete MTR geographic analysis is conditional on obtaining and validating station coordinates.

The minimum viable deliverable will contain:

1. district-level bus-stop and route metrics;
2. absolute, per-capita, and area-normalized comparisons;
3. linked maps and statistical charts; and
4. at least two hypothesis-driven findings.

MTR station and barrier-free analysis will be included after the coordinate checkpoint is passed. System-wide patronage trends and full route-line animation are optional extensions rather than core requirements.

The project will describe districts as having **relatively lower measured provision**, not as definitively underserved. The available data do not directly measure service frequency, vehicle capacity, operating hours, walking time, crowding, reliability, or individual travel needs.

## Risks and Mitigation

| Risk | Mitigation |
|---|---|
| MTR station coordinates are unavailable or cannot be joined reliably | Complete the bus analysis first; use MTR only after station IDs or names are validated |
| Bus-route geometry is too large for an interactive browser map | Simplify geometry, filter selected routes, or display district aggregates |
| Duplicate route-stop records inflate counts | Count unique stop IDs and apply a documented route-ID rule |
| Census and transport datasets refer to different years | Display reference dates and treat comparisons as cross-sectional context |
| District size and population produce conflicting rankings | Present absolute, per-capita, and area-based measures together |
| Correlations are misleading because there are only 18 districts | Show all observations, identify outliers, and avoid causal claims |

## Task List, Timeline, and Division of Labour

| Period | Task | Responsible member(s) | Output |
|---|---|---|---|
| 7-10 October | Download bus datasets; locate and validate MTR station coordinates; audit fields and dates | Ryan | Data-readiness checkpoint |
| 11-16 October | Clean census and transport data; define route and station counting rules | Ryan | Reproducible cleaned datasets |
| 17-23 October | Perform district spatial joins and calculate indicators | Ryan | District-level analysis table |
| 24 October-6 November | Build maps, comparison charts, and linked interactions | Ryan | Preliminary dashboard |
| 7-13 November | Test hypotheses, investigate outliers, and revise visual encodings | Ryan | Analytical findings |
| 14-20 November | Optimize dashboard and prepare the live walkthrough | Ryan and group members | Demo-ready product |
| 21-29 November | Document methods, limitations, and findings; package final files | Ryan and group members | Final report and work product |

The public-transport member will lead data validation, spatial processing, visualization, hypothesis testing, and documentation for this sector. Shared dashboard integration and visual consistency will be coordinated with the other group members.

## References

Census and Statistics Department. (n.d.). *2016 Population By-census: Statistics and boundaries of District Council districts* [Data set]. DATA.GOV.HK. https://data.gov.hk/en-data/dataset/hk-censtatd-census_geo-2016-population-bycensus-by-dcd

Census and Statistics Department. (n.d.). *2021 Population Census: Statistics and boundaries of District Council districts* [Data set]. DATA.GOV.HK. https://data.gov.hk/tc-data/dataset/hk-censtatd-census_geo-2021-population-census-by-dcd

Home Affairs Department. (n.d.). *Hong Kong administrative boundaries* [Data set]. DATA.GOV.HK. https://data.gov.hk/en-data/dataset/hk-had-json1-hong-kong-administrative-boundaries

MTR Corporation Limited. (n.d.). *MTR routes, fares and barrier-free facilities* [Data set]. DATA.GOV.HK. https://data.gov.hk/en-data/dataset/mtr-data-routes-fares-barrier-free-facilities

Transport Department. (n.d.). *Bus route* [Data set]. Common Spatial Data Infrastructure Portal. https://static.csdi.gov.hk/csdi-webpage/download/7faa97a82780505c9673c4ba128fbfed/geojson

Transport Department. (n.d.). *Coordinate of bus stop location* [Data set]. Common Spatial Data Infrastructure Portal. https://static.csdi.gov.hk/csdi-webpage/download/6a20951f9a1f5c1d981e80d8a45d141c/geojson

Transport Department. (n.d.). *Monthly Traffic and Transport Digest* [Data set]. DATA.GOV.HK. https://data.gov.hk/en-data/dataset/hk-td-tis_10-monthly-traffic-and-transport-digest

Transport Department. (n.d.). *Routes and fares of public transport (GeoJSON)* [Data set]. DATA.GOV.HK. https://data.gov.hk/tc-data/dataset/hk-td-tis_23-routes-fares-geojson