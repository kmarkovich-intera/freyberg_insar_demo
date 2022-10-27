#script to run proof of concept for inSAR data assimilation in gw decision support modeling

#build pst for complex model, add csub pars, draw one realization, and run it.
# the heads/flows will be assimilated in the simple model and recharge will be the truth (spatial and temporal)

#first build pst to do traditional DA with heads

#then build pst to do traditional DA with heads and streamflow obs
# we should have a better constraint on R/K, but not necessarily on spatial/temporal recharge

#then add insar values to DA and show spatial/temporal advantage

import os
import re
import platform
import codecs
import sys
import time
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import flopy
import flopy.discretization as fgrid
from flopy.utils.zonbud import ZoneBudget
import flopy.plot as fplot
from flopy.utils.gridintersect import GridIntersect
import descartes
import shapely
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.strtree import STRtree
import shapefile
import pyemu
import shutil
import math
from collections import OrderedDict
import seaborn as sns
import string
from matplotlib.gridspec import GridSpec

keep = ['arrobs_head_k:0_i:13_j:10', 'arrobs_head_k:2_i:2_j:9', 'arrobs_head_k:2_i:33_j:7', 'sfr_usecol:gage_1']
keep_labels = ["gw_1","gw_2","gw_3","sw_1"]
keep_units = ["$m$","$m$","$m$","$\\frac{m^3}{d}$"]
keep_dict = {k:l for k,l in zip(keep,keep_labels)}
keep_dict2 = {k:l for k,l in zip(keep,keep_units)}

forecast = ["sfr_usecol:tailwater","sfr_usecol:headwater","arrobs_head_k:0_i:9_j:1"]
forecast_labels = ["tailwater sw-gw exchg","headwater sw-gw exchg","gw forecast"]
forecast_dict = {k:l for k,l in zip(forecast,forecast_labels)}
forecast_units = ["$\\frac{m^3}{d}$","$\\frac{m^3}{d}$","$m$"]
forecast_dict2 = {k:l for k,l in zip(forecast,forecast_units)}


def prep_deps(d):
    """copy exes to a directory based on platform

    Args:

        d (str): directory to copy into
    Note:
        currently only mf6 for mac and win and mp7 for win are in the repo.
            need to update this...

    """
    # copy in deps and exes
    if "window" in platform.platform().lower():
        bd = os.path.join("executables", "windows")
    elif "linux" in platform.platform().lower():
        bd = os.path.join("executables", "linux")
    else:
        bd = os.path.join("executables", "mac")
    dest_dirs = [dd for dd in os.listdir(d) if
                 os.path.isdir(os.path.join(d, dd)) and "inset" in dd.lower()]

    for f in os.listdir(bd):
        if not f.startswith('pestpp'):
            for dd in dest_dirs:
                shutil.copy2(os.path.join(bd, f), os.path.join(d, dd, f))
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))
        else:
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))
    for src_d in ["flopy", "pyemu"]:
        assert os.path.exists(src_d), src_d
        dest_d = os.path.join(d, src_d)
        if os.path.exists(dest_d):
            shutil.rmtree(dest_d)
        shutil.copytree(src_d, dest_d)


def prep_truth_model(t_d, run=False):

    new_d = t_d.replace('_org', '_sub')

    if os.path.exists(new_d):
        shutil.rmtree(new_d)

    # # fix the wel flux files
    # wel_files = [f for f in os.listdir(t_d) if ".wel_stress_period_data" in f]
    # for wel_file in wel_files:
    #     sp = wel_file.split('_')[-1].split('.')[0]
    #     df = pd.read_csv(os.path.join(t_d, wel_file), header=None, names=["l", "r", "c", "flux"])
    #     df = df.astype({"l": int, "r": int, "c": int, "flux": np.float16})
    #     df.to_csv(os.path.join(t_d, "freyberg6.wel_stress_period_data_{0}.txt".format(sp)), index=False,
    #               header=False,  sep=" ")

    #load model
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d)
    t = sim.get_model()
    nper = sim.tdis.nper.array
    prd = sim.tdis.perioddata.array
    nlay, nrow, ncol = t.dis.nlay.array, t.dis.nrow.array, t.dis.ncol.array
    hobs = t.head_obs.continuous.get_data()
    sobs = t.sfr_obs.continuous.get_data()

    sim_name = 'freyberg_csub'

    new_sim = flopy.mf6.MFSimulation(
        sim_name=sim_name, sim_ws=new_d, continue_=True)

    flopy.mf6.ModflowTdis(
        new_sim, nper=nper, perioddata=prd,
    )
    flopy.mf6.ModflowIms(
        new_sim,
        complexity = 'complex',
    )
    gwf = flopy.mf6.ModflowGwf(
        new_sim, modelname=sim_name, save_flows=True, newtonoptions="newton", print_flows=False, )

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=t.dis.delr.array,
        delc=t.dis.delc.array,
        top=t.dis.top.array,
        botm=t.dis.botm.array,
        idomain=t.dis.idomain.array,
    )

    flopy.mf6.ModflowGwfic(gwf, strt=t.ic.strt.array)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=[1,0,0],
        k=[3,.3,30],
        k33 = [.3,.03,3]
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=[1,0,0],
        ss=0.0,
        sy=[0.1,0.25,0.1],
        transient={0: False, 1: True},
        save_flows=True,
    )

    #rech is sine wave inverse to pumping
    fs = 732  # sample rate
    f = 2  # the frequency of the signal
    x = np.arange(fs)  # the points on the x axis for plotting
    # compute the value (amplitude) of the sin wave at the for each sample
    y = 0.0001*np.sin(2 * np.pi * f * (x / fs)) + 0.0001

    rec_data = {}
    files = [f for f in os.listdir(t_d) if ".rch" in f.lower() and f.endswith(".txt")]
    assert len(files) > 0
    for i in range(len(files)):
        arr = np.ones((nrow,ncol))
        arr = arr * y[i]
        rec_data[i] = arr

    rch = flopy.mf6.ModflowGwfrcha(
        gwf,
        recharge=rec_data,
        save_flows=True,
    )

    ghb = flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data = t.ghb.stress_period_data.get_data(),
        save_flows=True,
    )


    obs = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs",
        filename="{}.obs".format(sim_name),
        continuous=hobs,
    )

    # compute the value (amplitude) of the sin wave at the for each sample
    y = 150*np.sin(2 * np.pi * f * (x / fs)) - 300

    rec_data = {}
    for kper,ra in t.wel.stress_period_data.data.items():
        df = pd.DataFrame.from_records(ra)
        df.q = y[kper]
        rec_data[kper] = df.values.tolist()

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data = rec_data,
        save_flows=True,
    )

    y = 250 * np.sin(2 * np.pi * f * (x / fs)) + 500

    rec_data = {}

    for kper in range(nper):
        rec_data[kper] = [[0,'inflow',y[kper]]]


    sfr = flopy.mf6.ModflowGwfsfr(
        gwf,
        unit_conversion=86400.,
        nreaches=120,
        packagedata=t.sfr.packagedata.get_data(),
        perioddata=rec_data,
        connectiondata=t.sfr.connectiondata.get_data(),
        save_flows=True,
        obs_filerecord='sfr.obs',
        boundnames=True
    )

    sfr.obs.initialize(
        filename="sfr.obs",
        continuous=sobs,
    )

    # inputs for csub
    gammaw = 9806.65  # Compressibility of water (Newtons/($m^3$)
    beta = 4.6612e-10  # Specific gravity of water (1/$Pa$)
    sgm_str = "1.77, 1.60, 1.77"  # Specific gravity of moist soils (unitless)
    sgs_str = "2.06, 1.94, 2.06"  # Specific gravity of saturated soils (unitless)
    cg_theta_str = "0.32, 0.45, 0.32"  # Coarse-grained material porosity (unitless)
    cg_ske_str = "0.00005, 0.0003, 0.00005"  # Elastic specific storage ($1/m$)
    # ssv_cv_str = "0.0003, 0.0003, 0.0003"  #  initial inelastic specific storage ($1/m$)
    # sse_cr_str = "0.00005, 0.0003, 0.00005" # initial Elastic specific storage ($1/m$)
    ib_thick_str = "1., 2.5, 1."  # Interbed thickness ($m$)
    ib_theta = 0.45  # Interbed initial porosity 0.0003(unitless)
    ib_cr = 0.01  # Interbed recompression index (unitless)
    ib_cv = 0.25  # Interbed compression index (unitless)
    stress_offset = 15.0  # Initial preconsolidation stress offset ($m$)

    # parse strings into tuples
    sgm = [float(value) for value in sgm_str.split(",")]
    sgs = [float(value) for value in sgs_str.split(",")]
    cg_theta = [float(value) for value in cg_theta_str.split(",")]
    cg_ske = [float(value) for value in cg_ske_str.split(",")]
    # ssv_cv = [float(value) for value in ssv_cv_str.split(",")]
    # sse_cr = [float(value) for value in sse_cr_str.split(",")]
    ib_thick = [float(value) for value in ib_thick_str.split(",")]


    # create interbed package data
    icsubno = 0
    csub_pakdata = []
    for i in range(nrow):
        for j in range(ncol):
            for k in range(nlay):
                if t.dis.idomain.array[k,i,j] == 0:
                    continue
                else:
                    boundname = "{:02d}_{:02d}_{:02d}".format(k + 1, i + 1, j + 1)
                    ib_lst = [
                        icsubno,
                        (k, i, j),
                        "nodelay",
                        stress_offset,
                        ib_thick[k],
                        1.0,
                        # ssv_cv[k],
                        # sse_cr[k],
                        ib_cv,
                        ib_cr,
                        ib_theta,
                        0.001,
                        0,
                        boundname,
                    ]
                    csub_pakdata.append(ib_lst)
                    icsubno += 1

    oc = flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord="truth.cbb",
            head_filerecord="truth.hds",
            saverecord=[("HEAD", "ALL"),("BUDGET", "ALL")],
        )


    sub = flopy.mf6.ModflowGwfcsub(
            gwf,
            # print_input=True,
            save_flows=True,
            head_based = False,
            compaction_filerecord = 'truth.cmp',
            compaction_elastic_filerecord = 'truth.cec',
            compaction_inelastic_filerecord = 'truth.cnc',
            compaction_interbed_filerecord = 'truth.cic',
            compaction_coarse_filerecord = 'truth.ccc',
            zdisplacement_filerecord = 'truth.zbz',
            specified_initial_preconsolidation_stress=False,
            compression_indices=True,
            boundnames=True,
            ninterbeds=len(csub_pakdata),
            sgm=sgm,
            sgs=sgs,
            cg_theta=cg_theta,
            cg_ske_cr=cg_ske,
            beta=beta,
            gammaw=gammaw,
            packagedata=csub_pakdata,
        )


    new_sim.set_all_data_external()
    new_sim.write_simulation()
    prep_deps(new_d)

    if run:
        pyemu.os_utils.run("mf6", cwd=new_d)

def setup_run_truth_pst(t_d, num_reals = 100):
    template_ws = "tmp_pst_daily"
    temp_model_ws = "temp"
    if os.path.exists(temp_model_ws):
        shutil.rmtree(temp_model_ws)
    shutil.copytree(t_d, temp_model_ws)
    prep_deps(temp_model_ws)

    base_model_ws = None

    # load flow model
    flow_dir = os.path.join(temp_model_ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name="mf6")
    m = sim.get_model("freyberg_csub")

    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=temp_model_ws, new_d=template_ws,
                             remove_existing=True,
                             longnames=True, spatial_reference=m.modelgrid,
                             zero_based=False, start_datetime="1-1-2022")

    prep_deps(template_ws)

    flow_dts = pd.to_datetime("1-1-2022") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")


    tmp_files = []
    files = [os.path.join(os.path.split(template_ws)[-1], f) for f in os.listdir(template_ws) if
             f.split(".")[-1] in ["hds", "cbb", "cec", "ccc", "cic","cnc","zbz"]]
    files.sort()
    for f in files:
        pf.tmp_files.append(f)

    # add obs
    df = pd.read_csv(os.path.join(template_ws, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", index_cols=['time'], use_cols=list(df.columns), obsgp="hds",
                        prefix="hds", ofile_sep=",")

    # pf.add_py_function('workflow.py', 'postprocess()', is_pre_cmd=False)

    df = pd.read_csv(os.path.join(template_ws, "sfr.csv"), index_col=0)
    pf.add_observations("sfr.csv", index_cols=["time"], use_cols=df.columns.tolist(), ofile_sep=",",
                        obsgp="flow", prefix="sfr")

    # # add observations for the simulated rech values
    # pf.add_py_function("workflow.py","extract_rech()", is_pre_cmd=False)
    # test_extract_rech(template_ws)
    # prefix = "rech_k:1"
    # pf.add_observations('rech.txt', prefix=prefix, obsgp=prefix)

    # add observations for simulated hds states
    pf.add_py_function("workflow.py", "extract_z_disp_obs()", is_pre_cmd=False)
    fnames = test_extract_z_disp_obs(template_ws)
    for k, fname in enumerate(fnames):
        prefix = "zdisp_k:{0}".format(k)
        pf.add_observations(fname, prefix=prefix, obsgp=prefix)

    # for tag in ["hds"]:
    #     arrs = [f for f in os.listdir(template_ws) if f.startswith(tag) and f.endswith(".txt")]
    #     arrs.sort()
    #     print(arrs)
    #     for arr in arrs:
    #         pf.add_observations(arr, prefix=arr.split('.')[0].replace("_", ""),
    #                             obsgp=arr.split('.')[0].replace("_", ""))

    # the geostruct object for grid-scale parameters
    grid_v = pyemu.geostats.ExpVario(contribution=1.0, a=500)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)

    # the geostruct object for pilot-point-scale parameters
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=2000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v)

    # the geostruct for recharge grid-scale parameters
    rch_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    rch_gs = pyemu.geostats.GeoStruct(variograms=rch_v)

    # the geostruct for temporal correlation
    temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=30))

    pp_cells = 5

    tags = {"npf_k_": [0.2, 5., 1e-7, 500],
            "npf_k33_": [0.2, 5, 1e-7, 500],
            "sto_sy": [0.5, 2, 0.01, 0.25],
            "recharge_": [0.5, 2, 0, 0.1],
            "cg_ske_": [0.1, 10., 0.000001, 0.001],
            "cg_theta_": [0.25, 1.75, 0.01, 0.3]}

    # tags = {"cg_ske_": [0.1, 10., 0.000001, 0.001],
    #         "cg_theta_": [0.25, 1.75, 0.01, 0.3]}
    #interbed thickess, inelastic Ss, interbed porosity are all conductance style pars (not array)

    # use the idomain array for masking parameter locations
    try:
        ib = m.dis.idomain[0].array
    except:
        ib = m.dis.idomain[0]

    # loop over each tag, bound info pair
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        # find all array based files that have the tag in the name
        arr_files = [f for f in os.listdir(template_ws) if tag in f and f.endswith(".txt")]

        if len(arr_files) == 0:
            print("warning: no array files found for ", tag)
            continue

        # make sure each array file in nrow X ncol dimensions (not wrapped)
        for arr_file in arr_files:
            print(arr_file)
            arr = np.loadtxt(os.path.join(template_ws, arr_file)).reshape(ib.shape)
            np.savetxt(os.path.join(template_ws, arr_file), arr, fmt="%15.6E")

        # if this is the recharge tag
        if "rch" in tag:
            # add one set of grid-scale parameters for all files
            pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                              pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                              geostruct=rch_gs)

            # add one constant parameter for each array, and assign it a datetime
            # so we can work out the temporal correlation
            for arr_file in arr_files:
                arr = np.loadtxt(os.path.join(template_ws, arr_file))
                print(arr_file,arr.mean(),arr.std())
                uub = arr.mean() * ub
                llb = arr.mean() * lb
                if "daily" in t_d.lower():
                    uub *= 5
                    llb /= 5
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp="rch_const", zone_array=ib, upper_bound=uub, lower_bound=llb,
                                  par_style="direct")

        # otherwise...
        else:
            # for each array add both grid-scale and pilot-point scale parameters
            for arr_file in arr_files:
                pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                                  pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=grid_gs)
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=grid_gs)
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints",
                                  par_name_base=arr_file.split('.')[1] + "_pp",
                                  pargp=arr_file.split('.')[1] + "_pp", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  pp_space=pp_cells, geostruct=pp_gs)


    # get all the list-type files associated with the wel package
    list_files = [f for f in os.listdir(t_d) if "freyberg_csub.wel_stress_period_data_" in f and f.endswith(".txt")]
    # for each wel-package list-type file
    for list_file in list_files:
        kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        # add spatially constant, but temporally correlated parameter
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=5, lower_bound=0.2, datetime=flow_dts[kper])

    lb,ub = .2,5
    if "daily" in t_d.lower():
        lb, ub = .1, 10
    # add temporally indep, but spatially correlated grid-scale parameters, one per well
    pf.add_parameters(filenames=list_files, par_type="grid", par_name_base="wel_grid",
                      pargp="wel_grid", index_cols=[0, 1, 2], use_cols=[3],
                      upper_bound=ub, lower_bound=lb)

    # add grid-scale parameters for SFR reach conductance.  Use layer, row, col and reach
    # number in the parameter names
    pf.add_parameters(filenames="freyberg_csub.sfr_packagedata.txt", par_name_base="sfr_rhk",
                      pargp="sfr_rhk", index_cols=[0, 1, 2, 3], use_cols=[9], upper_bound=20.,
                      lower_bound=0.05, par_type="grid")

    # SFR inflow
    files = [f for f in os.listdir(template_ws) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    print(files)
    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        # add the parameters
        pf.add_parameters(filenames=f,
                          index_cols=[0],  # reach number
                          use_cols=[2],  # columns with parameter values
                          par_type="grid",
                          par_name_base="sfrgr",
                          pargp="sfrgr",
                          upper_bound=10, lower_bound=0.1,  # don't need ult_bounds because it is a single multiplier
                          datetime=flow_dts[kper],  # this places the parameter value on the "time axis"
                          geostruct=temporal_gs)

    # # add grid-scale parameters for CSUB interbed thickness
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_thk",
    #                   pargp="csub_thk", index_cols=[0, 1, 2, 3], use_cols=[6], upper_bound=2.,
    #                   lower_bound=0.5, par_type="grid", ult_ubound=10., ult_lbound=0.1)

    # add grid-scale parameters for CSUB interbed porosity
    pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_ibt",
                      pargp="csub_ibt", index_cols=[0, 1, 2, 3], use_cols=[10], upper_bound=2.,
                      lower_bound=0.5, par_type="grid", ult_ubound=.45, ult_lbound=0.01)

    # # add grid-scale parameters for CSUB interbed inelastic Ss
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_ssv",
    #                   pargp="csub_ssv", index_cols=[0, 1, 2, 3], use_cols=[8], upper_bound=10.,
    #                   lower_bound=0.1, par_type="grid", ult_ubound=0.01, ult_lbound=0.00001)
    #
    # # add grid-scale parameters for CSUB interbed elastic Ss
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_sse",
    #                   pargp="csub_sse", index_cols=[0, 1, 2, 3], use_cols=[9], upper_bound=10.,
    #                   lower_bound=0.1, par_type="grid", ult_ubound=0.001, ult_lbound=0.000001)

    # add grid-scale parameters for CSUB interbed K33
    pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_kv",
                      pargp="csub_kv", index_cols=[0, 1, 2, 3], use_cols=[11], upper_bound=10.,
                      lower_bound=0.1, par_type="grid", ult_ubound=0.1, ult_lbound=0.00001)


    # add model run command
    pf.mod_sys_cmds.append("mf6")

    pf.extra_py_imports.append("shutil")
    pf.extra_py_imports.append("time")
    pf.extra_py_imports.append("flopy")
    pf.extra_py_imports.append("platform")

    # build pest control file
    pst = pf.build_pst('freyberg.pst')
    par = pst.parameter_data

    # #tie cg_ske and sse_cr together
    # cpar = par.loc[par.parnme.str.contains("cg_ske"), :]
    # spar = par.loc[par.parnme.str.contains("sse"), :]
    # vpar = par.loc[par.parnme.str.contains("ssv"), :]
    # print(cpar, spar, vpar)
    # exit()

    # #tie sse_cr and ssv_cc together
    # rpar = par.loc[par.parnme.str.contains("recharge"), :]

    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)
    pe.enforce()
    pe.to_binary(os.path.join(template_ws,"prior_pe.jcb"))

    # write the control file
    pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)

    # run with noptmax = 0
    pyemu.os_utils.run("{0} freyberg.pst".format(
        os.path.join("pestpp-ies")), cwd=pf.new_d)

    # make sure it ran
    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)

    # define what file has the prior parameter ensemble
    pst.pestpp_options["ies_par_en"] = "prior_pe.jcb"
    pst.pestpp_options["save_binary"] = True


    #now draw a single realization and run
    pst.parameter_data.loc[:, "parval1"] = pe.loc[pe.index[0], pst.par_names].values
    pst.control_data.noptmax = 0
    pst.write(os.path.join(template_ws, "freyberg.pst"))
    pyemu.os_utils.run("pestpp-ies freyberg.pst", cwd=template_ws)

def build_localizer(t_d):
    #hds and streamflow can't inform porosity pars

    obs = pst.observation_data.loc[pst.nnz_obs_names, :]
    hobs = obs.loc[obs.oname == "hds", "obsnme"].values
    sobs = obs.loc[obs.oname == "flow", "obsnme"].values
    assert hobs.shape[0] > 0

    pargps = par.pargp.unique()
    pargps.sort()
    tpar_tags = ["csub_ibt", "cg_theta"]
    tpargps = []
    for pargp in pargps:
        is_t = False
        for tag in tpar_tags:
            if tag in pargp:
                is_t = True
                break
        if is_t:
            tpargps.append(pargp)
    print(tpargps)

    df = pd.DataFrame(columns=pargps, index=pst.nnz_obs_names, dtype=float)
    df.loc[:, :] = 1.0
    df.loc[hobs, tpargps] = 0.0
    df.loc[sobs, tpargps] = 0.0

    print(df.shape)
    assert df.shape[0] == pst.nnz_obs
    assert df.shape[1] == len(pargps)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    df.sum(axis=0).plot.bar(ax=ax, alpha=0.5)
    plt.tight_layout()
    plt.savefig("localizer.pdf")

    plt.close(fig)
    df.to_csv(os.path.join(t_d, "localizer.csv"))
    pst.pestpp_options["ies_localizer"] = "localizer.csv"
    pst.write(os.path.join(t_d, "freyberg.pst"), version=2)

def map_complex_to_simple_bat(c_d,b_d,real_idx):
    """map the daily model outputs to the monthly model observations

    Args:
        c_d (str): the daily (complex) model prior monte carlo master dir
        b_d (str): the monthly (simple) batch model tmeplate dir
        real_idx (int): the complex model prior monte carlo obs ensemble index position
            to use as the observations for the monthly model

    """

    # load the daily model obs ensemble and get the simulated values
    coe = pd.read_csv(os.path.join(c_d, "freyberg.0.obs.csv"))
    coe.set_index('real_name')
    cvals = coe.loc[real_idx,:]

    # load the batch monthly model control file
    bpst = pyemu.Pst(os.path.join(b_d,"freyberg.pst"))
    obs = bpst.observation_data
    is_1_lay = True
    if True in [True if "k:2" in o else False for o in bpst.obs_names]:
        is_1_lay = False

    # assign all common observations
    if is_1_lay:
        #idx = cvals.index.copy()
        for k in keep:
            if "k:2" not in k:
                continue
            d = cvals.index.map(lambda x: k not in x)
            cvals = cvals.loc[d]
            kk = k.replace("k:2","k:0")
            cvals.index = cvals.index.map(lambda x: x.replace("k:2","k:0") if kk in x else x)

        for k in forecast:
            if "k:2" not in k:
                continue
            d = cvals.index.map(lambda x: k not in x)
            cvals = cvals.loc[d]
            kk = k.replace("k:2", "k:0")
            cvals.index = cvals.index.map(lambda x: x.replace("k:2", "k:0") if kk in x else x)
    print(cvals)
    d = set(bpst.obs_names) - set(cvals.index.tolist())
    print(d)
    obs.loc[:,"obsval"] = cvals.loc[bpst.obs_names]
    assert obs.obsval.shape[0] == obs.obsval.dropna().shape[0]
    assert cvals.loc[bpst.obs_names].shape[0] == cvals.loc[bpst.obs_names].dropna().shape[0]

    # set some weights - only the first 12 months (not counting spin up time)
    # just picked weights that seemed to balance-ish the one realization I looked at...
    obs.loc[:,"weight"] = 0.0
    for k in keep.copy():
        if is_1_lay:
            k = k.replace("k:2","k:0")
        kobs = obs.loc[obs.obsnme.str.contains(k),:].copy()
        kobs.loc[:,"time"] = kobs.time.apply(float)
        kobs.sort_values(by="time",inplace=True)
        if "gage" in k:
            obs.loc[kobs.obsnme[1:13], "weight"] = 0.005
        else:
            obs.loc[kobs.obsnme[1:13],"weight"] = 2.0

    # setup a template dir for this complex model realization
    t_d = b_d
    bpst.write(os.path.join(t_d,"freyberg.pst"),version=2)
    return t_d

def run_ies(tmp_ws, m_d="master_ies", num_workers=10, num_reals=30, noptmax=-1, drop_conflicts=True,
                port=4263, hostname=None, subset_size=4, bad_phi_sigma=1000.0, overdue_giveup_fac=None,
                use_condor=False, overdue_giveup_minutes=None):
        bat_dir = os.path.join(tmp_ws)
        # ies stuff
        pst = pyemu.Pst(os.path.join(tmp_ws, "lisbon.pst"))
        pst.pestpp_options["ies_num_reals"] = num_reals
        pst.pestpp_options["ies_drop_conflicts"] = drop_conflicts
        pst.pestpp_options["ies_subset_size"] = subset_size
        pst.pestpp_options["ies_bad_phi_sigma"] = bad_phi_sigma
        if overdue_giveup_fac is not None:
            pst.pestpp_options["overdue_giveup_fac"] = overdue_giveup_fac
        if overdue_giveup_minutes is not None:
            pst.pestpp_options["overdue_giveup_minutes"] = overdue_giveup_minutes
        pst.control_data.noptmax = noptmax
        pst.write(os.path.join(bat_dir, "lisbon.pst"), version=2)
        prep_worker(tmp_ws, tmp_ws + "_clean")

        # run ies
        # added master_dir=m_d as workaround for both local and cluster runs. hopefully no further need to use the subprocess.Popen(args)
        master_p = None
        local = True
        if hostname is None:  # or use_condor: # this means we run ies local, not with condor
            pyemu.os_utils.start_workers(tmp_ws + "_clean", "pestpp-ies", "lisbon.pst", num_workers=num_workers,
                                         worker_root=".",
                                         port=port, local=local, master_dir=m_d)
        elif use_condor:
            jobid = condor_submit(template_ws=tmp_ws + "_clean", pstfile="lisbon.pst", conda_zip='condorpy38.tar.gz',
                                  subfile='lisbon.sub',
                                  workerfile='worker.sh', executables=['mf6', 'pestpp-ies'], request_memory=4000,
                                  request_disk='10g', port=port,
                                  num_workers=num_workers)

            # jwhite - commented this out so not starting local workers on the condor submit machine # no -ross
            pyemu.os_utils.start_workers(tmp_ws + "_clean", "pestpp-ies", "lisbon.pst", num_workers=0, worker_root=".",
                                         port=port, local=local, master_dir=m_d)

            if jobid is not None:
                # after run master is finished clean up condor by using condor_rm
                print(f'killing condor job {jobid}')
                os.system(f'condor_rm {jobid}')

        # if a master was spawned, wait for it to finish
        if master_p is not None:
            master_p.wait()

def invest():

    # cbb = flopy.utils.binaryfile.CellBudgetFile(os.path.join('daily_model_files_sub','truth.cbb'))
    # cbb = cbb.list_records()

    cmp = flopy.utils.binaryfile.HeadFile(os.path.join('daily_model_files_sub','truth.cmp'), text='CSUB-COMPACTION')
    # zbz = cmp.list_records()

    zbz = flopy.utils.binaryfile.HeadFile(os.path.join('daily_model_files_sub','truth.zbz'), text='CSUB-ZDISPLACE')
    # zbz = zbz.list_records()
    zbz = zbz.get_alldata()

    # plt.imshow(zbz[-1,0,:,:])
    # plt.show()

    plt.plot(zbz[:,0,10,10])
    # zbz[zbz>1000]='nan'
    # zbz[zbz < 0] = 'nan'
    # zbz = np.log10(zbz)
    # plt.imshow(zbz[300,0,:,:])
    plt.show()


    # with open(os.path.join('daily_model_files_sub','CSub_Delay.csub_z_dis'), 'rb') as f:
    #     contents = f.read()
    #
    # print(contents)

def extract_z_disp_obs():
    hds = flopy.utils.HeadFile('truth.zbz', text='CSUB-ZDISPLACE')
    arr = hds.get_data()
    fnames = []
    for i in range(1,len(arr)-1):
        arr[i] = arr[i] - arr[i-1]
    arr[0] = 0.

    for k, a in enumerate(arr):
        fname = 'zdisp_' + str(k) + '.dat'
        np.savetxt(fname, a, fmt='%15.6E')
        fnames.append(fname)
    return fnames

def test_extract_z_disp_obs(t_d):
    cwd = os.getcwd()
    os.chdir(t_d)
    fnames = extract_z_disp_obs()
    os.chdir(cwd)
    return fnames

def extract_rech():
    HK_lay1 = np.loadtxt('freyberg_csub.npf_k_layer1.txt')
    try:
        HK_lay2 = np.loadtxt('freyberg_csub.npf_k_layer2.txt')
    except:
        print('no layer 2')
    try:
        HK_lay3 = np.loadtxt('freyberg_csub.npf_k_layer3.txt')
    except:
        print('no layer 3')
    try:
        HK_lay1 = (HK_lay1 + HK_lay2 + HK_lay3)/3
    except:
        print('just a single layer model in a multilayer world')

    np.savetxt('rech.txt', HK_lay1, fmt='%15.6E')

def test_extract_rech(t_d):
    cwd = os.getcwd()
    os.chdir(t_d)
    fnames = extract_rech()
    os.chdir(cwd)
    return fnames

def prep_simple_model(t_d, run=False):

    new_d = t_d.replace('_org', '_sub')

    if os.path.exists(new_d):
        shutil.rmtree(new_d)

    # # # fix the wel flux files
    # wel_files = [f for f in os.listdir(t_d) if ".wel_stress_period_data" in f]
    # for wel_file in wel_files:
    #     sp = wel_file.split('_')[-1].split('.')[0]
    #     df = pd.read_csv(os.path.join(t_d, wel_file), header=None, names=["l", "r", "c", "flux"])
    #     df = df.astype({"l": int, "r": int, "c": int, "flux": np.float16})
    #     df.to_csv(os.path.join(t_d, "freyberg6.wel_stress_period_data_{0}.txt".format(sp)), index=False,
    #               header=False,  sep=" ")


    #load model
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d)
    t = sim.get_model()
    nper = sim.tdis.nper.array
    prd = sim.tdis.perioddata.array
    nlay, nrow, ncol = t.dis.nlay.array, t.dis.nrow.array, t.dis.ncol.array
    hobs = t.head_obs.continuous.get_data()
    sobs = t.sfr_obs.continuous.get_data()

    sim_name = 'freyberg_csub'

    new_sim = flopy.mf6.MFSimulation(
        sim_name=sim_name, sim_ws=new_d,
    )
    flopy.mf6.ModflowTdis(
        new_sim, nper=nper, perioddata=prd,
    )
    flopy.mf6.ModflowIms(
        new_sim,
        complexity = 'complex',continue_=True,
    )
    gwf = flopy.mf6.ModflowGwf(
        new_sim, modelname=sim_name, save_flows=True, newtonoptions="newton", print_flows=False,
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=t.dis.delr.array,
        delc=t.dis.delc.array,
        top=t.dis.top.array,
        botm=t.dis.botm.array,
        idomain=t.dis.idomain.array,
    )

    flopy.mf6.ModflowGwfic(gwf, strt=t.ic.strt.array)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=t.npf.icelltype.array,
        k=t.npf.k.array/10,
        k33 = t.npf.k33.array/10
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=t.npf.icelltype.array,
        ss=0.0,
        sy=t.sto.sy.array,
        transient={0: False, 1: True},
        save_flows=True,
    )

    rec_data = {}
    files = [f for f in os.listdir(t_d) if ".rch" in f.lower() and f.endswith(".txt")]
    assert len(files) > 0
    for f in files:
        arr = np.ones((nrow, ncol)) * 0.00001
        sp = int(f.split(".")[1].split("_")[-1])
        rec_data[(sp - 1)] = arr

    rch = flopy.mf6.ModflowGwfrcha(
        gwf,
        recharge = rec_data,
        save_flows=True,
    )

    ghb = flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data = t.ghb.stress_period_data.get_data(),
        save_flows=True,
    )


    obs = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs",
        filename="{}.obs".format(sim_name),
        continuous=hobs,
    )


    rec_data = {}
    for kper,ra in t.wel.stress_period_data.data.items():
        df = pd.DataFrame.from_records(ra)
        df.q = -300
        rec_data[kper] = df.values.tolist()

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data = rec_data,
        save_flows=True,
    )

    rec_data = {}

    for kper in range(nper):
        rec_data[kper] = [[0,'inflow',500]]

    sfr = flopy.mf6.ModflowGwfsfr(
        gwf,
        unit_conversion=86400.,
        nreaches=40,
        packagedata=t.sfr.packagedata.get_data(),
        perioddata=rec_data,
        connectiondata=t.sfr.connectiondata.get_data(),
        save_flows=True,
        obs_filerecord='sfr.obs',
        boundnames=True
    )

    sfr.obs.initialize(
        filename="sfr.obs",
        continuous=sobs,
    )

    # inputs for csub
    gammaw = 9806.65  # Compressibility of water (Newtons/($m^3$)
    beta = 4.6612e-10  # Specific gravity of water (1/$Pa$)
    sgm_str = "1.77, 1.60, 1.77"  # Specific gravity of moist soils (unitless)
    sgs_str = "2.06, 1.94, 2.06"  # Specific gravity of saturated soils (unitless)
    cg_theta_str = "0.32, 0.45, 0.32"  # Coarse-grained material porosity (unitless)
    cg_ske_str = "0.00005, 0.0003, 0.00005"  # Elastic specific storage ($1/m$)
    # ssv_cv_str = "0.0003, 0.0003, 0.0003"  #  initial inelastic specific storage ($1/m$)
    # sse_cr_str = "0.00005, 0.0003, 0.00005" # initial Elastic specific storage ($1/m$)
    ib_thick_str = "1., 2.5, 1."  # Interbed thickness ($m$)
    ib_theta = 0.45  # Interbed initial porosity 0.0003(unitless)
    ib_cr = 0.01  # Interbed recompression index (unitless)
    ib_cv = 0.25  # Interbed compression index (unitless)
    stress_offset = 15.0  # Initial preconsolidation stress offset ($m$)

    # parse strings into tuples
    sgm = [float(value) for value in sgm_str.split(",")]
    sgs = [float(value) for value in sgs_str.split(",")]
    cg_theta = [float(value) for value in cg_theta_str.split(",")]
    cg_ske = [float(value) for value in cg_ske_str.split(",")]
    # ssv_cv = [float(value) for value in ssv_cv_str.split(",")]
    # sse_cr = [float(value) for value in sse_cr_str.split(",")]
    ib_thick = [float(value) for value in ib_thick_str.split(",")]

    # create interbed package data
    icsubno = 0
    csub_pakdata = []
    for i in range(nrow):
        for j in range(ncol):
            for k in range(nlay):
                if t.dis.idomain.array[k, i, j] == 0:
                    continue
                else:
                    boundname = "{:02d}_{:02d}_{:02d}".format(k + 1, i + 1, j + 1)
                    ib_lst = [
                        icsubno,
                        (k, i, j),
                        "nodelay",
                        stress_offset,
                        ib_thick[k],
                        1.0,
                        # ssv_cv[k],
                        # sse_cr[k],
                        ib_cv,
                        ib_cr,
                        ib_theta,
                        0.001,
                        0,
                        boundname,
                    ]
                    csub_pakdata.append(ib_lst)
                    icsubno += 1

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="truth.cbb",
        head_filerecord="truth.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sub = flopy.mf6.ModflowGwfcsub(
        gwf,
        # print_input=True,
        save_flows=True,
        head_based=False,
        compaction_filerecord='truth.cmp',
        compaction_elastic_filerecord='truth.cec',
        compaction_inelastic_filerecord='truth.cnc',
        compaction_interbed_filerecord='truth.cic',
        compaction_coarse_filerecord='truth.ccc',
        zdisplacement_filerecord='truth.zbz',
        specified_initial_preconsolidation_stress=False,
        compression_indices=True,
        boundnames=True,
        ninterbeds=len(csub_pakdata),
        sgm=sgm,
        sgs=sgs,
        cg_theta=cg_theta,
        cg_ske_cr=cg_ske,
        beta=beta,
        gammaw=gammaw,
        packagedata=csub_pakdata,
    )


    new_sim.set_all_data_external()
    new_sim.write_simulation()
    prep_deps(new_d)

    if run:
        pyemu.os_utils.run("mf6", cwd=new_d)

def prep_worker(org_d, new_d):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)
    exts = ["jcb", "rei", "dbf", "shp", "shx"]

    files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
    for f in files:
        if f != 'prior.jcb' and f != 'lisbon.4.par.jcb' and f != 'lisbon.0.dv_pop_frntload.jcb':  # need prior.jcb to run ies and posterior ensemble for mou
            os.remove(os.path.join(new_d, f))
    mlt_dir = os.path.join(new_d, "mult")
    for f in os.listdir(mlt_dir)[1:]:
        os.remove(os.path.join(mlt_dir, f))
    tpst = os.path.join(new_d, "temp.pst")
    if os.path.exists(tpst):
        os.remove(tpst)

def condor_submit(template_ws, pstfile, conda_zip='geopy38.tar.gz', subfile='condor.sub', workerfile='worker.sh',
                  executables=[], request_memory=4000, request_disk='10g', port=4200, num_workers=71):
    """
    :param template_ws: path to template_ws
    :param pstfile: name of pest control file
    :param conda_zip: conda-pack zip file
    :param subfile: condor submit file name
    :param workerfile: condor worker file name
    :param executables: any executables in the template_ws that might need permissions changed
    :param request_memory: memory to request for each job
    :param request_disk: disk space
    :param port: port number, should be same as the one used when running the master
    :param num_workers: number of workers to start
    :return:
    """
    # template_ws = os.path.join('model_ws', 'template')

    if not os.path.join(conda_zip):
        str = f'conda-pack dir {conda_zip} does not exist\n ' + f'consider running conda-pack while in your conda env\n'
        AssertionError(str)
    conda_base = conda_zip.replace(".tar.gz", "")

    # should probably remove to remove tmp files to make copying faster...

    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'temp')):
        shutil.rmtree(os.path.join(cwd, 'temp'))
    shutil.copytree(os.path.join(cwd, template_ws), 'temp')

    # zip template_ws
    os.system(f'tar cfvz temp.tar.gz temp')

    if not os.path.exists('log'):
        os.makedirs('log')

    # write worker file
    worker_f = open(os.path.join(cwd, workerfile), 'w')
    worker_lines = ['#!/bin/sh\n',
                    '\n',
                    '# make conda-pack dir\n',
                    f'mkdir {conda_base}\n',
                    f'tar -xf {conda_zip} -C {conda_base}\n',
                    '\n',
                    '# unzip temp\n',
                    'tar xzf temp.tar.gz\n',
                    'cd temp\n',
                    '\n',
                    '# add python to path (relative)\n',
                    f'export PATH=../{conda_base}/bin:$PATH\n',
                    'python -c "print(\'python is working\')"\n',
                    'which python',
                    '\n']

    if len(executables) > 0:
        worker_lines += ['# make sure executables have permissions\n'] + [f'chmod +x {exe}\n' for exe in executables]

    worker_lines += ['\n',
                     f'./pestpp-ies {pstfile} /h $1:$2\n']
    worker_f.writelines(worker_lines)
    worker_f.close()

    sub_f = open(os.path.join(cwd, subfile), 'w')
    sublines = ['# never ever change this!\n',
                'notification = Never\n',
                '\n',
                "# just plain'ole vanilla for us!\n",
                'universe = vanilla\n',
                '\n',
                '# this will log all the worker stdout and stderr - make sure to mkdir a "./log" dir where ever\n',
                '# the condor_submit command is issued\n',
                'log = log/worker_$(Cluster).log\n',
                'output = log/worker_$(Cluster)_$(Process).out\n',
                'error = log/worker_$(Cluster)_$(Process).err\n',
                '\n', '# define what system is required\n',
                'requirements = ( (OpSys == "LINUX") && (Arch == "X86_64"))\n',
                '# how much mem each worker needs in mb\n',
                f'request_memory = {request_memory}\n',
                '\n',
                '# how many cpus per worker\n',
                'request_cpus = 1\n',
                '\n',
                '# how much disk space each worker needs in gb (append a "g")\n',
                f'request_disk = {request_disk}\n',
                '\n',
                '# the command to execute remotely on the worker hosts to start the condor "job"\n',
                f'executable = {workerfile}\n',
                '\n',
                '# the command line args to pass to worker.sh.  These are the 0) IP address/UNC name of the master host\n',
                '# and 1) the port number for pest comms.  These must in that order as they are used in worker.sh\n',
                '# ausdata-head1.cluster or 10.99.10.30 \n',
                f'arguments = ausdata-head1.cluster {port}\n',
                '\n',
                '# stream the info back to the log files\n',
                'stream_output = True\n',
                'stream_error = True\n',
                '\n',
                '# transfer the files to start the job\n',
                'should_transfer_files = YES\n',
                'when_to_transfer_output = ON_EXIT\n',
                '\n',
                '# the files to transfer before starting the job (in addition to the executable command file)\n',
                f'transfer_input_files = temp.tar.gz, {conda_zip}\n',
                '\n',
                '# number of workers to start\n',
                f'queue {num_workers}']
    sub_f.writelines(sublines)
    sub_f.close()

    os.system(f'condor_submit {subfile} > condor_jobID.txt')

    jobfn = open('condor_jobID.txt')
    lines = jobfn.readlines()
    jobfn.close()
    jobid = lines[1].split()[-1].replace('.', '')
    print(f'{num_workers} job(s) submitted to cluster {jobid}.')

    return int(jobid)

def setup_simple_pst(t_d, hds=True, strmflw = False, zdisp=False, num_reals = 100,):

    if hds:
        new_d = "tmp_pst_hds"
    if strmflw:
        new_d = new_d + "_sfr"
    if zdisp:
        new_d = new_d + "_zdsp"

    # template_ws = "tmp_pst"
    temp_model_ws = "temp"
    if os.path.exists(temp_model_ws):
        shutil.rmtree(temp_model_ws)
    shutil.copytree(t_d, temp_model_ws)
    prep_deps(temp_model_ws)

    base_model_ws = None

    # load flow model
    flow_dir = os.path.join(temp_model_ws)
    sim = flopy.mf6.MFSimulation.load(sim_ws=flow_dir, exe_name="mf6")
    m = sim.get_model("freyberg_csub")

    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=temp_model_ws, new_d=new_d,
                             remove_existing=True,
                             longnames=True, spatial_reference=m.modelgrid,
                             zero_based=False, start_datetime="1-1-2022")

    prep_deps(new_d)

    flow_dts = pd.to_datetime("1-1-2022") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")

    tmp_files = []
    files = [os.path.join(os.path.split(new_d)[-1], f) for f in os.listdir(new_d) if
             f.split(".")[-1] in ["hds", "cbb", "cec", "ccc", "cic", "cnc", "zbz"]]
    files.sort()
    for f in files:
        pf.tmp_files.append(f)

    # add obs
    df = pd.read_csv(os.path.join(new_d, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", index_cols=['time'], use_cols=list(df.columns), obsgp="hds",
                        prefix="hds", ofile_sep=",")

    # pf.add_py_function('workflow.py', 'postprocess()', is_pre_cmd=False)

    df = pd.read_csv(os.path.join(new_d, "sfr.csv"), index_col=0)
    pf.add_observations("sfr.csv", index_cols=["time"], use_cols=df.columns.tolist(), ofile_sep=",",
                        obsgp="flow", prefix="sfr")

    # # add observations for the simulated rech values
    # pf.add_py_function("workflow.py","extract_rech()", is_pre_cmd=False)
    # test_extract_rech(template_ws)
    # prefix = "rech_k:1"
    # pf.add_observations('rech.txt', prefix=prefix, obsgp=prefix)

    # add observations for simulated hds states
    pf.add_py_function("workflow.py", "extract_z_disp_obs()", is_pre_cmd=False)
    fnames = test_extract_z_disp_obs(new_d)
    for k, fname in enumerate(fnames):
        prefix = "zdisp_k:{0}".format(k)
        pf.add_observations(fname, prefix=prefix, obsgp=prefix)

    # for tag in ["hds"]:
    #     arrs = [f for f in os.listdir(template_ws) if f.startswith(tag) and f.endswith(".txt")]
    #     arrs.sort()
    #     print(arrs)
    #     for arr in arrs:
    #         pf.add_observations(arr, prefix=arr.split('.')[0].replace("_", ""),
    #                             obsgp=arr.split('.')[0].replace("_", ""))

    # the geostruct object for grid-scale parameters
    grid_v = pyemu.geostats.ExpVario(contribution=1.0, a=500)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)

    # the geostruct object for pilot-point-scale parameters
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=2000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v)

    # the geostruct for recharge grid-scale parameters
    rch_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    rch_gs = pyemu.geostats.GeoStruct(variograms=rch_v)

    # the geostruct for temporal correlation
    temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=30))

    pp_cells = 5

    tags = {"npf_k_": [0.2, 5., 1e-7, 500],
            "npf_k33_": [0.2, 5, 1e-7, 500],
            "sto_sy": [0.5, 2, 0.01, 0.25],
            "recharge_": [0.5, 2, 0, 0.1],
            "cg_ske_": [0.1, 10., 0.000001, 0.001],
            # "ssv_cv":  [0.1, 10., 0.0001, 0.01], #these are conductance-style pars
            # "ib_theta_": [0.25, 1.75, 0.10, 0.45], #these are conductance-style pars
            # ib_thick_
            "cg_theta_": [0.25, 1.75, 0.01, 0.3]}
    # interbed thickess, inelastic Ss, interbed porosity are all conductance style pars (not array)

    # use the idomain array for masking parameter locations
    try:
        ib = m.dis.idomain.array
    except:
        ib = m.dis.idomain

    # loop over each tag, bound info pair
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        # find all array based files that have the tag in the name
        arr_files = [f for f in os.listdir(new_d) if tag in f and f.endswith(".txt")]

        if len(arr_files) == 0:
            print("warning: no array files found for ", tag)
            continue

        # make sure each array file in nrow X ncol dimensions (not wrapped)
        for arr_file in arr_files:
            print(arr_file)
            arr = np.loadtxt(os.path.join(new_d, arr_file)).reshape(ib.shape)
            np.savetxt(os.path.join(new_d, arr_file), arr, fmt="%15.6E")

        # if this is the recharge tag
        if "rch" in tag:
            # add one set of grid-scale parameters for all files
            pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                              pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                              geostruct=rch_gs)

            # add one constant parameter for each array, and assign it a datetime
            # so we can work out the temporal correlation
            for arr_file in arr_files:
                arr = np.loadtxt(os.path.join(new_d, arr_file))
                print(arr_file, arr.mean(), arr.std())
                uub = arr.mean() * ub
                llb = arr.mean() * lb
                if "daily" in t_d.lower():
                    uub *= 5
                    llb /= 5
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp="rch_const", zone_array=ib, upper_bound=uub, lower_bound=llb,
                                  par_style="direct")

        # otherwise...
        else:
            # for each array add both grid-scale and pilot-point scale parameters
            for arr_file in arr_files:
                pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                                  pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=grid_gs)
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=grid_gs)
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints",
                                  par_name_base=arr_file.split('.')[1] + "_pp",
                                  pargp=arr_file.split('.')[1] + "_pp", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  pp_space=pp_cells, geostruct=pp_gs)

    # get all the list-type files associated with the wel package
    list_files = [f for f in os.listdir(t_d) if "freyberg_csub.wel_stress_period_data_" in f and f.endswith(".txt")]
    # for each wel-package list-type file
    for list_file in list_files:
        kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        # add spatially constant, but temporally correlated parameter
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=5, lower_bound=0.2, datetime=flow_dts[kper])

    lb, ub = .2, 5
    if "daily" in t_d.lower():
        lb, ub = .1, 10
    # add temporally indep, but spatially correlated grid-scale parameters, one per well
    pf.add_parameters(filenames=list_files, par_type="grid", par_name_base="wel_grid",
                      pargp="wel_grid", index_cols=[0, 1, 2], use_cols=[3],
                      upper_bound=ub, lower_bound=lb)

    # add grid-scale parameters for SFR reach conductance.  Use layer, row, col and reach
    # number in the parameter names
    pf.add_parameters(filenames="freyberg_csub.sfr_packagedata.txt", par_name_base="sfr_rhk",
                      pargp="sfr_rhk", index_cols=[0, 1, 2, 3], use_cols=[9], upper_bound=20.,
                      lower_bound=0.05, par_type="grid")

    # SFR inflow
    files = [f for f in os.listdir(new_d) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    print(files)
    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        # add the parameters
        pf.add_parameters(filenames=f,
                          index_cols=[0],  # reach number
                          use_cols=[2],  # columns with parameter values
                          par_type="grid",
                          par_name_base="sfrgr",
                          pargp="sfrgr",
                          upper_bound=10, lower_bound=0.1,  # don't need ult_bounds because it is a single multiplier
                          datetime=flow_dts[kper],  # this places the parameter value on the "time axis"
                          geostruct=temporal_gs)

    # # add grid-scale parameters for CSUB interbed thickness
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_thk",
    #                   pargp="csub_thk", index_cols=[0, 1, 2, 3], use_cols=[6], upper_bound=2.,
    #                   lower_bound=0.5, par_type="grid", ult_ubound=10., ult_lbound=0.1)

    # add grid-scale parameters for CSUB interbed porosity
    pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_ibt",
                      pargp="csub_ibt", index_cols=[0, 1, 2, 3], use_cols=[10], upper_bound=2.,
                      lower_bound=0.5, par_type="grid", ult_ubound=.45, ult_lbound=0.01)

    # # add grid-scale parameters for CSUB interbed inelastic Ss
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_ssv",
    #                   pargp="csub_ssv", index_cols=[0, 1, 2, 3], use_cols=[8], upper_bound=10.,
    #                   lower_bound=0.1, par_type="grid", ult_ubound=0.01, ult_lbound=0.00001)
    #
    # # add grid-scale parameters for CSUB interbed elastic Ss
    # pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_sse",
    #                   pargp="csub_sse", index_cols=[0, 1, 2, 3], use_cols=[9], upper_bound=10.,
    #                   lower_bound=0.1, par_type="grid", ult_ubound=0.001, ult_lbound=0.000001)

    # add grid-scale parameters for CSUB interbed K33
    pf.add_parameters(filenames="freyberg_csub.csub_packagedata.txt", par_name_base="csub_kv",
                      pargp="csub_kv", index_cols=[0, 1, 2, 3], use_cols=[11], upper_bound=10.,
                      lower_bound=0.1, par_type="grid", ult_ubound=0.1, ult_lbound=0.00001)

    # add model run command
    pf.mod_sys_cmds.append("mf6")

    pf.extra_py_imports.append("shutil")
    pf.extra_py_imports.append("time")
    pf.extra_py_imports.append("flopy")
    pf.extra_py_imports.append("platform")

    # build pest control file
    pst = pf.build_pst('freyberg.pst')
    par = pst.parameter_data

    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)
    pe.enforce()
    pe.to_binary(os.path.join(new_d, "prior_pe.jcb"))

    # write the control file
    pst.write(os.path.join(pf.new_d, "freyberg.pst"), version=2)

    # run with noptmax = 0
    pyemu.os_utils.run("{0} freyberg.pst".format(
        os.path.join("pestpp-ies")), cwd=pf.new_d)

    # make sure it ran
    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)

    # define what file has the prior parameter ensemble
    pst.pestpp_options["ies_par_en"] = "prior_pe.jcb"
    pst.pestpp_options["save_binary"] = True

    #now for some obs and weights!
    pst.observation_data.weight = 0

    map_complex_to_simple_bat('tmp_pst_daily',new_d, 0)
    obs = pst.observation_data

    if strmflw==False:
        kobs = obs.loc[obs.obsnme.str.contains('gage'), :].copy()
        kobs.loc[:, "time"] = kobs.time.apply(float)
        kobs.sort_values(by="time", inplace=True)
        obs.loc[kobs.obsnme, "weight"] = 0.0

    if zdisp==False:
        kobs = obs.loc[obs.obsnme.str.contains('zdisp'), :].copy()
        kobs.loc[:, "time"] = kobs.time.apply(float)
        kobs.sort_values(by="time", inplace=True)
        obs.loc[kobs.obsnme, "weight"] = 0.0

    pst.write(os.path.join(pf.new_d,"lisbon.pst"),version=2)

    build_localizer(pf.new_d)

if __name__ == "__main__":
    # invest()

    ##prep model and pst
    # prep_truth_model('daily_model_files_org', run=True)
    # prep_simple_model('monthly_model_files_org', run=True)

    ## run truth model
    # setup_run_truth_pst('daily_model_files_sub', num_reals=10)

    ##run first scenario
    setup_simple_pst('monthly_model_files_sub', hds=True)
    run_ies('tmp_pst_hds', m_d="master_ies_hds", num_workers=8, num_reals=200, noptmax=3, drop_conflicts=True,
                port=4263, hostname=None, subset_size=4, bad_phi_sigma=1000.0, overdue_giveup_fac=10,
                use_condor=False)
    #postprocess

    ##run second scenario
    # setup_simple_pst('monthly_model_files_sub', hds=True, strmflw=True)
    # run_ies('tmp_pst_hds_sfr', m_d="master_ies_hd_sfr", num_workers=8, num_reals=200, noptmax=3, drop_conflicts=True,
    #             port=4263, hostname=None, subset_size=4, bad_phi_sigma=1000.0, overdue_giveup_fac=10,
    #             use_condor=False)


    ##run third scenario
    # setup_simple_pst('monthly_model_files_sub', hds=True, strmflw=True, zdisp=True)
    # run_ies('tmp_pst_hds_sfr_zdsp', m_d="master_ies_hds_sfr_zdsp", num_workers=8, num_reals=200, noptmax=3, drop_conflicts=True,
    #             port=4263, hostname=None, subset_size=4, bad_phi_sigma=1000.0, overdue_giveup_fac=10,
    #             use_condor=False)