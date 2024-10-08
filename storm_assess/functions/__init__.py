"""
Provides example functions that are useful for assessing model 
tropical storms.


"""
import numpy
import datetime
import calendar

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom

import iris
import iris.coord_systems as icoord_systems
import iris.coords as icoords


#: Lat/lon locations for each ocean basin for mapping. If set to 
#: None then it returns a global map
MAP_REGION = {'na': (-105, 0, 60, 0),
              'ep': (-170, -80, 40, 0),
              'wp': (-265, -180, 50, 0),
              'ni': (-310, -260, 30, 0),
              'si': (-340, -260, 0, -40),
              'au': (-270, -195, 0, -40),
              'sp': (-200, -100, 0, -40),
              'sa': ( -90, 0, 0, -40),
              'nh': (-360, 30, 70, 0),
              'sh': (-360, 30, 0, -90),
              None: (-360, 0, 90, -90)
              }

#: Lat/lon locations of tracking regions for each ocean basin. If None
#: then returns a region for the whole globe
TRACKING_REGION = {'na': ([-75, -20, -20, -80, -80, -100, -100, -75, -75], [0, 0, 60, 60, 40, 40, 20, 6, 0]),
                   'ep': ([-140, -75, -75, -100, -100, -140, -140], [0, 0, 6, 20, 30, 30, 0]),
                   #'wp': ([-260, -180, -180, -260, -260], [0, 0, 60, 60, 0]),
                   'wp': ([-260, -180, -180, -260, -260], [0, 0, 30, 30, 0]),
                   'cp': ([-180, -140, -140, -180, -180], [0, 0, 50, 50, 0]),
                   'ni': ([-320, -260, -260, -320, -320], [0, 0, 30, 30, 0]),
                   'si': ([-330, -270, -270, -330, -330], [-40, -40, 0, 0, -40]),
                   'au': ([-270, -200, -200, -270, -270], [-40, -40, 0, 0, -40]),
                   'sp': ([-200, -120, -120, -200, -200], [-40, -40, 0, 0, -40]),
                   'sa': ([-90, 0, 0, -90, -90], [-40, -40, 0, 0, -40]),
#                   'nh': ([-360, 0, 0, -360, -360],[0, 0, 90, 90 ,0]),
                   'nh': ([-359.9, 0, 0, -359.9, -359.9],[0, 0, 90, 90 ,0]),
#                   'sh': ([-360, 0, 0, -360, -360],[-90, -90, 0, 0 ,-90]),
                   'sh': ([-359.9, 0, 0, -359.9, -359.9],[-90, -90, 0, 0 ,-90]),
                   'mdr': ([-80, -20, -20, -80, -80], [10, 10, 20, 20, 10]),
                   None: ([-360, 0, 0, -360, -360],[-90, -90, 90, 90 ,-90])
                   }    

#: Corresponding full basin names for each abbreviation
BASIN_NAME = {'na': 'North Atlantic',
              'ep': 'Eastern Pacific',
              'wp': 'Western Pacific',
              'cp': 'Central Pacific',
              'ni': 'North Indian Ocean',
              'si': 'Southwest Indian Ocean',
              'au': 'Australian Region',
              'sp': 'South Pacific',
              'nh': 'Northern Hemisphere',
              'sh': 'Southern Hemisphere',
              None: 'Global'
              }

#: Corresponding month name for a given integer value
NUM_TO_MONTH = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep',10: 'Oct',11: 'Nov',12: 'Dec'}


def _get_time_range(year, months):
    """ 
    Creates a start and end date (a datetime.date timestamp) for a 
    given year and a list of months. If the list of months overlaps into 
    the following year (for example [11,12,1,2,3,4]) then the end date 
    adds 1 to the original year 
    
    """
    start_date = datetime.datetime(year, months[0], 1)
    end_year = year
    end_month = months[-1]+1
    if months[-1]+1 < months[0] or months[-1]+1 == 13 or len(months) >= 12:
        end_year = year+1
    if months[-1]+1 == 13:
        end_month = 1
    end_date = datetime.datetime(end_year, end_month, 1)
    return start_date, end_date
                
        
def _storms_in_time_range(storms, year, months):
    """Returns a generator of storms that formed during the desired time period """
    start_date, end_date = _get_time_range(year, months)
    for storm in storms:        
        if (storm.genesis_date() >= start_date) and (storm.genesis_date() < end_date):
            yield storm


def load_map(basin=None):
    """ Produces map for desired ocean basins for plotting. """ 
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-160))
    if basin == None:
        ax.set_global()
    else:
        ax.set_extent(MAP_REGION.get(basin))
    resolution ='50m' # use '10m' for fine scale and '110m' for  coarse scale (default)    
#    plt.gca().coastlines() 
#    land_feature = cfeature.NaturalEarthFeature('physical','land',resolution,edgecolor='none',facecolor=cfeature.COLORS['land_alt1'])
#    ax.add_feature(land_feature)
#    img = plt.imread('/home/h05/hadjn/ne_shaded_high_res.PNG')
#    ax.imshow(img[::-1, :], origin='lower', transform=ccrs.PlateCarree(), 
#                extent=[-180, 180, -90, 90], regrid_shape=(2000, 1000))
    return ax
    
    
def _basin_polygon(basin, project=True):
    """ 
    Returns a polygon of the tracking region for a particular 
    ocean basin. i.e. a storm must pass through this region 
    in order to be retined. For example, if basin is set to 
    'au' then storms for the Australian region must pass
    through the area defined by -270 to -200W, 0 to -40S.
    
    """
    rbox = sgeom.Polygon(zip(*TRACKING_REGION.get(basin)))
    if project: 
        rbox = ccrs.PlateCarree().project_geometry(rbox, ccrs.PlateCarree())
    return rbox
    
    
def _storm_in_basin(storm, basin):
    """ Returns True if a storm track intersects a defined ocean basin """
    rbox = _basin_polygon(basin)   
    lons, lats = zip(*[(ob.lon, ob.lat) for ob in storm.obs])
    track = sgeom.LineString(zip(lons, lats))       
    projected_track = ccrs.PlateCarree().project_geometry(track, ccrs.Geodetic())
    if rbox.intersects(projected_track):
        return True
    return False


def _get_genesis_months(storms, years, basin):
    """ 
    Returns genesis month of all storms that formed within a 
    given set of years 
    
    """
    genesis_months = []
    for storm in storms:
        if (storm.genesis_date().year in years) and _storm_in_basin(storm, basin):
            genesis_months.append(storm.genesis_date().month)
    return genesis_months
            
            
def _get_monthly_storm_count(storms, years, months, basin):
    """ Returns list of storm counts for a desired set of months """
    genesis_months = _get_genesis_months(storms, years, basin)
    monthly_count = []
    for month in months:
        monthly_count.append(genesis_months.count(month))
    return monthly_count


def _month_names(months):
    """ Returns list of month names for a given set of integer values """
    names = []
    for month in months:
        names.append(NUM_TO_MONTH.get(month))
    return names
        
        
def _get_time_period(years, months):
    """ 
    Returns string of time period for a given set of 
    years and months. E.g. months [6,7,8,9] and years 
    numpy.arange(1989,2003) would return a string 
    'June-September 1989-2002'. Note: years and 
    months must be lists or arrays.
    
    
    """    
    start_mon = calendar.month_name[months[0]]
    end_mon = calendar.month_name[months[::-1][0]]
    start_yr = str(years.min())
    end_yr = str(years.max())
    if start_yr == end_yr:
        return '%s-%s %s' % (start_mon, end_mon, start_yr)
    else:
        return '%s-%s %s-%s' % (start_mon, end_mon, start_yr, end_yr)
    
    
def _cube_data(data):
    """Returns a cube given a list of lat lon information."""
    cube = iris.cube.Cube(data)
    lat_lon_coord_system = icoord_systems.GeogCS(6371229)
    
    step = 4.0
    start = step/2
    count = 90
    pts = start + numpy.arange(count, dtype=numpy.float32) * step
    lon_coord = icoords.DimCoord(pts, standard_name='longitude', units='degrees', 
                                 coord_system = lat_lon_coord_system, circular=True)
    lon_coord.guess_bounds()
    
    start = -90
    step = 4.0
    count = 45
    pts = start + numpy.arange(count, dtype=numpy.float32) * step
    lat_coord = icoords.DimCoord(pts, standard_name='latitude', units='degrees', 
                                 coord_system = lat_lon_coord_system)
    lat_coord.guess_bounds()
    
    cube.add_dim_coord(lat_coord, 0)
    cube.add_dim_coord(lon_coord, 1)
    return cube 


def _binned_cube(lats, lons):
    """ Returns a cube (or 2D histogram) of lat/lons locations. """   
    data = numpy.zeros(shape=(45,90))
    binned_cube = _cube_data(data)
    xs, ys = binned_cube.coord('longitude').contiguous_bounds(), binned_cube.coord('latitude').contiguous_bounds()
    binned_data, _, _ = numpy.histogram2d(lons, lats, bins=[xs, ys])
    binned_cube.data = numpy.transpose(binned_data)
    return binned_cube

    
def storm_lats_lons(storms, years, months, basin, genesis=False, 
                 lysis=False, max_intensity=False):
    """ 
    Returns array of latitude and longitude values for all storms that 
    occurred within a desired year, month set and basin. 
    
    To get genesis, lysis or max intensity results set:
    Genesis plot: genesis=True
    Lysis plot: lysis=True
    Maximum intensity (location of max wind): 
    max_intensity=True
    
    """
    lats, lons = [], []
    count = 0
    for year in years:
        for storm in _storms_in_time_range(storms, year, months):
            if _storm_in_basin(storm, basin):
                if genesis:
                    #print 'getting genesis locations'
                    lats.extend([storm.obs_at_genesis().lat])
                    lons.extend([storm.obs_at_genesis().lon])
                elif lysis:
                    #print 'getting lysis locations'
                    lats.extend([storm.obs_at_lysis().lat])
                    lons.extend([storm.obs_at_lysis().lon])
                elif max_intensity:
                    #print 'getting max int locations'
                    lats.extend([storm.obs_at_vmax().lat])
                    lons.extend([storm.obs_at_vmax().lon])
                else:
                    #print 'getting whole storm track locations'
                    lats.extend([ob.lat for ob in storm.obs])
                    lons.extend([ob.lon for ob in storm.obs])
                count += 1
                
    # Normalise lon values into the range 0-360
    norm_lons = []
    for lon in lons:
        norm_lons.append((lon + 720) % 360)
    return lats, norm_lons, count


def get_projected_track(storm, map_proj):
    """ Returns track of storm as a linestring """
    lons, lats = zip(*[(ob.lon, ob.lat) for ob in storm.obs])
    track = sgeom.LineString(zip(lons, lats))
    projected_track = map_proj.project_geometry(track, ccrs.Geodetic())
    return projected_track
