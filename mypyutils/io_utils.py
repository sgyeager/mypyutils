import xarray as xr
import numpy as np
import glob
from functools import partial
import sys


def file_dict(filetempl,filetype,mem,stmon):
    """ returns a dictionary of filepaths keyed by initialization year, 
    for a given experiment, field, ensemble member, and initialization month
    """
    
    memstr = '{0:03d}'.format(mem)
    monstr = '{0:02d}'.format(stmon)
    filepaths = {}
    
    filetemp = filetempl.replace('MM',monstr).replace('EEE',memstr)

    #find all the relevant files
    files = sorted(glob.glob(filetemp))
        
    for file in files:
        #isolate initialization year from the file name
        ystr = file.split(filetype)[0]
        y0 = int(ystr[-11:-7])
        filepaths[y0]=file
        
    return filepaths

def nested_file_list_by_year(filetemplate,filetype,ens,field,startyears,stmon):
    """ retrieves a nested list of files for these start years and ensemble members
    """
    #ens = np.array(range(ens))+1
    yrs = startyears
    files = []    # a list of lists, dim0=start_year, dim1=ens
    filecount = []
    for yy,i in zip(yrs,range(len(yrs))):
        ffs = []  # a list of files for this yy
        file0 = ''
        first = True
        for ee in ens:
            filepaths = file_dict(filetemplate,filetype,ee,stmon)
            #append file if it is new
            if yy in filepaths.keys():
                file = filepaths[yy]
                if file != file0:
                    ffs.append(file)
                    file0 = file

        #append this ensemble member to files
        if ffs:  #only append if you found files
            files.append(ffs)
            filecount.append(1)
        else:
            filecount.append(0)
    return files,yrs[filecount!=0]

def get_monthly_data(filetemplate,filetype,ens,nlead,field,startyears,stmon,preproc,chunks={}):
    """ returns a dask array containing the requested hindcast ensemble
    """

    file_list,yrs = nested_file_list_by_year(filetemplate,filetype,ens,field,startyears,stmon)

    ds0 = xr.open_mfdataset(
    file_list,
    combine="nested",
    # concat_dim depends on how file_list is ordered; 
    # inner most list of datasets is combined along "M"; 
    # then the outer list is combined along "Y"
    concat_dim=["Y","M"],
    parallel=True,
    data_vars=[field],
    coords="minimal",
    compat="override",
    preprocess=partial(preproc,nlead=nlead,field=field),chunks=chunks)

    # assign final attributes
    ds0["Y"] = startyears
    ds0["M"] = np.arange(ds0.sizes["M"]) + 1

    # reorder into desired format (Y,L,M,...)
    ds0 = ds0.transpose("Y","L","M",...)
    return ds0
