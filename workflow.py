#script to run proof of concept for inSAR data assimilation in gw decision support modeling

#build pst for complex model, add csub pars, draw one realization, and run it.
# the heads will be assimilated in the simple model and recharge will be the truth (spatial and temporal)

#first build pst to do traditional DA with heads and lower K
# (it should adjust the tmp recharge pars lower, and maybe also adjust spatial rech)

#then build pst to do traditional DA with heads and higher K
# (it should adjust the tmp recharge pars higher, and maybe also adjust spatial rech)

#then add insar values to DA and show spatial

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
    # wel_files = {int(f.split(".")[1].split("_")[-1]): pd.read_csv(os.path.join(t_d, f), header=None,
    #                                                               names=["l", "r", "c", "flux"]) for f in wel_files}
    #
    # for sp, df in wel_files.items():
    #     df.to_csv(os.path.join(t_d, "freyberg6.wel_stress_period_data_{0}.txt".format(sp)), index=False, header=False,
    #               sep=" ")

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
        complexity = 'complex',
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
        icelltype=[1,0,0],
        k=[3,.3,30],
        k33 = [.3,.03,3]
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=[1,0,0],
        ss=0.0,
        sy=0.02,
        transient={0: True},
        save_flows=True,
    )

    rec_data = {}
    files = [f for f in os.listdir(t_d) if ".rch" in f.lower() and f.endswith(".txt")]
    assert len(files) > 0
    for f in files:
        arr = np.loadtxt(os.path.join(t_d, f))
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

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data = t.wel.stress_period_data.get_data(),
        save_flows=True,
    )

    sfr = flopy.mf6.ModflowGwfsfr(
        gwf,
        unit_conversion=86400.,
        nreaches=120,
        packagedata=t.sfr.packagedata.get_data(),
        perioddata=t.sfr.perioddata.get_data(),
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
    cg_ske_str = "0.005, 0.01, 0.005"  # Elastic specific storage ($1/m$)
    ib_thick_str = "1., 1., 1."  # Interbed thickness ($m$)
    ib_theta = 0.45  # Interbed initial porosity (unitless)
    ib_cr = 0.01  # Interbed recompression index (unitless)
    ib_cv = 0.25  # Interbed compression index (unitless)
    stress_offset = 0.0  # Initial preconsolidation stress offset ($m$)

    # parse strings into tuples
    sgm = [float(value) for value in sgm_str.split(",")]
    sgs = [float(value) for value in sgs_str.split(",")]
    cg_theta = [float(value) for value in cg_theta_str.split(",")]
    cg_ske = [float(value) for value in cg_ske_str.split(",")]
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
                        ib_cv,
                        ib_cr,
                        ib_theta,
                        999.0,
                        999.0,
                        boundname,
                    ]
                    csub_pakdata.append(ib_lst)
                    icsubno += 1

    oc = flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord="truth.cbb",
            head_filerecord="truth.hds",
            saverecord=[("BUDGET", "ALL")],
        )


    sub = flopy.mf6.ModflowGwfcsub(
            gwf,
            # print_input=True,
            save_flows=True,
            compaction_filerecord = 'truth.cmp',
            compaction_elastic_filerecord = 'truth.cec',
            compaction_inelastic_filerecord = 'truth.cnc',
            compaction_interbed_filerecord = 'truth.cic',
            compaction_coarse_filerecord = 'truth.ccc',
            zdisplacement_filerecord = 'truth.zbz',
            compression_indices=True,
            update_material_properties=False,
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

def setup_run_truth_pst(t_d):
    template_ws = "tmp_pst"
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

    # add copy arrays and run call
    # pf.add_py_function("workflow.py", "convert_zdis_insar()", is_pre_cmd=False)

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

    # df = pd.read_csv(os.path.join(template_ws, "insar.csv"), index_col=0)
    # pf.add_observations("ltsm_flow.csv", index_cols=["datetime"],
    #                     use_cols=["fromzone0", "tozone0", "fromzone1", "tozone1"], ofile_sep=",",
    #                     obsgp="ltsm_flw")


    for tag in ["hds"]:
        arrs = [f for f in os.listdir(template_ws) if f.startswith(tag) and f.endswith(".txt")]
        arrs.sort()
        print(arrs)
        for arr in arrs:
            pf.add_observations(arr, prefix=arr.split('.')[0].replace("_", ""),
                                obsgp=arr.split('.')[0].replace("_", ""))

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
            "sto_sy_": [0.5, 2, 0.01, 0.25],
            "recharge_": [0.5, 2, 0, 0.1],
            "sgs_": [0.5, 2, 0.001, 100],
            "sgm_": [.01, 10, 0.001, 500.],
            "atv_": [0.5, 2, 0.005, 1],
            "cg_ske_": [0.5, 2.0, 0.0001, 0.1],
            "cg_theta_": [0.5, 2.0, 0.0001, 0.1]}

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
                if "daily" in org_ws.lower():
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
    list_files = [f for f in os.listdir(org_ws) if "freyberg6.wel_stress_period_data_" in f and f.endswith(".txt")]
    # for each wel-package list-type file
    for list_file in list_files:
        kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        # add spatially constant, but temporally correlated parameter
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=5, lower_bound=0.2, datetime=flow_dts[kper])

    lb,ub = .2,5
    if "daily" in org_ws.lower():
        lb, ub = .1, 10
    # add temporally indep, but spatially correlated grid-scale parameters, one per well
    pf.add_parameters(filenames=list_files, par_type="grid", par_name_base="wel_grid",
                      pargp="wel_grid", index_cols=[0, 1, 2], use_cols=[3],
                      upper_bound=ub, lower_bound=lb)

    # add grid-scale parameters for SFR reach conductance.  Use layer, row, col and reach
    # number in the parameter names
    pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt", par_name_base="sfr_rhk",
                      pargp="sfr_rhk", index_cols=[0, 1, 2, 3], use_cols=[9], upper_bound=20.,
                      lower_bound=0.05,
                      par_type="grid")

    # add model run command
    pf.mod_sys_cmds.append("mf6")

    # build pest control file
    pst = pf.build_pst('freyberg.pst')
    par = pst.parameter_data

    # draw from the prior and save the ensemble in binary format
    pe = pf.draw(num_reals, use_specsim=True)

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

    pst.control_data.noptmax = -1

    # define what file has the prior parameter ensemble
    pst.pestpp_options["ies_par_en"] = "prior.jcb"
    pst.pestpp_options["save_binary"] = True

    # write the updated pest control file
    pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)
    return template_ws

def build_localizer(t_d):
    #make sure obs inform the right temporal pars
    #assume all obs (heads, streamflow, and insar) can inform all static props
    pst = pyemu.Pst(os.path.join(t_d, "freyberg.pst"))
    par = pst.parameter_data.loc[pst.adj_par_names, :]
    obs = pst.observation_data.loc[pst.nnz_obs_names, :]
    hobs = obs.loc[obs.oname == "head", "obsnme"].values
    assert hobs.shape[0] > 0

    pargps = par.pargp.unique()
    pargps.sort()
    tpar_tags = ["sgs", "sgm", "cg_ske", "cg_theta"]
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
    # df.loc[:,"pfac"] = 0.0
    df.loc[hobs, tpargps] = 0.0

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
    pst.control_data.noptmax = -2
    pst.write(os.path.join(t_d, "lisbon.pst"), version=2)
    pyemu.os_utils.run("pestpp-ies lisbon.pst", cwd=t_d)

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

    cmp = flopy.utils.binaryfile.HeadFile(os.path.join('daily_model_files_sub','CSub_Delay.csub_cmpct'), text='CSUB-COMPACTION')
    # zbz = cmp.list_records()

    zbz = flopy.utils.binaryfile.HeadFile(os.path.join('daily_model_files_sub','truth.csub_z_dis'), text='CSUB-ZDISPLACE')
    # zbz = zbz.list_records()
    zbz = zbz.get_alldata()

    # plt.imshow(zbz[-1,0,:,:])
    # plt.show()

    plt.plot(zbz[:,0,10,10])
    plt.show()


    # with open(os.path.join('daily_model_files_sub','CSub_Delay.csub_z_dis'), 'rb') as f:
    #     contents = f.read()
    #
    # print(contents)


if __name__ == "__main__":
    # prep_truth_model('daily_model_files_org', run=True)
    setup_run_truth_pst('daily_model_files_sub')
    # invest()