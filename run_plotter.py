import argparse
import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import root_pandas
import uproot as upr
import plotting.parse_yaml as pyml
import plotting.plot_dmc_hist as pldmc
import qRC.syst.qRC_systematics as syst
from qRC.tmva.IdMVAComputer import helpComputeIdMva
from joblib import delayed, Parallel

def remove_duplicates(vars):
    mask = len(vars) * [True]
    for i in range(len(vars)):
        for j in range(i+1,len(vars)):
            if vars[i] == vars[j]:
                mask[j] = False

    return list(itertools.compress(vars,mask))

def chunked_loader(fpath,columns, **kwargs):
    fitt = pd.read_hdf(fpath, columns=columns, chunksize=10000, iterator=True, **kwargs)
    df = pd.concat(fitt, copy=False)

    return df

def make_unique_names(plt_list):
    
    mtl_list = len(plt_list) * [0]
    for i in range(len(plt_list)):
        mult = 0
        for j in range(i):
            if plt_list[i]['type'] == plt_list[j]['type'] and plt_list[i]['var'] == plt_list[j]['var']:
                mult += 1

        mtl_list[i] = mult
                
    for i in range(len(plt_list)):
        plt_list[i]['num'] = mtl_list[i]
                
    return plt_list

def make_vars(plot_dict,extra=[],extens=True):
    ret = []
    for dic in plot_dict:
        ret.append(dic['var'])
        if 'exts' in dic.keys() and extens:
            for ext in dic['exts']:
                ret.append(dic['var'] + ext)

    ret.extend(extra)

    return remove_duplicates(ret)

def check_vars(df, varrs):

    varmiss = len(varrs) * [False]
    for i, var in enumerate(varrs):
        if not var in df.columns:
            varmiss[i] = True

    return varmiss

def main(options):
    
    plot_dict = make_unique_names(pyml.yaml_parser(options.config)())
    #varrs = make_vars(plot_dict,['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'probeEtaWidth', 'probePhiWidth','probeScEnergy', 'probeSigmaRR', 'newULPhoIDcorrAllFinal']) #nominal vars
    varrs = make_vars(plot_dict,['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'probeEtaWidth', 'probePhiWidth','probeScEnergy', 'probeSigmaRR', 'probePhoIso03'])  #for flashgg. Need wrongly labelled eta+phi widths
    varrs_data = make_vars(plot_dict,['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'probeEtaWidth', 'probePhiWidth','probePhiWidth_Sc', 'probeEtaWidth_Sc', 'probePhoIso03','probeScEnergy', 'probeSigmaRR'],extens=False)


    if 'probePhoIso03_uncorr' in varrs: #fgg fixes. can leave in anyway
        varrs.pop(varrs.index('probePhoIso03_uncorr'))

    if 'probePhoIdMVA_xml' in varrs: #fgg fixes. can leave in anyway
        varrs.pop(varrs.index('probePhoIdMVA_xml'))

    if options.mc.split('.')[-1] == 'root':
    #used when checking flashgg output 
        if options.mc_tree is None:
            raise NameError('mc_tree has to be in options if a *.root file is used as input')
        df_mc = root_pandas.read_root(options.mc, options.mc_tree, columns=varrs) #too much mem?

        #df_mc_file = upr.open(options.mc)
        #df_mc_tree = df_mc_file[options.mc_tree]
        #df_mc = df_mc_tree.pandas.df(varrs)
        ##df_mc = df_mc_trees[options.mc_tree].pandas.df(varrs.query('probePt>@self.ptmin and probePt<@self.ptmax and probeScEta>@self.etamin and probeScEta<@self.etamax and probePhi>@self.phimin and probePhi<@self.phimax')
    else:
        df_mc = pd.read_hdf(options.mc, columns=varrs)

    if options.data.split('.')[-1] == 'root':
    #used when checking flashgg output 
        if options.data_tree is None:
            raise NameError('data_tree has to be in options if a *.root file is used as input')
        df_data = root_pandas.read_root(options.data, options.data_tree, columns=varrs_data) #too much mem?
        #df_data_trees = upr.open(options.data)
        #df_data = df_data_trees[options.data_tree].pandas.df(varrs_data)
    else:
        df_data = pd.read_hdf(options.data, columns=varrs_data)
  
    #print 'mc columns: {}'.format(df_mc.columns)
    #print 'data columns: {}'.format(df_data.columns)

    if 'weight_clf' not in df_mc.columns and not options.no_reweight:
        print 'Doing 4D mc reweighting...'
        if options.reweight_cut is not None:
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=options.reweight_cut)
        else:
            warnings.warn('Cut for reweighting is taken from 0th and 1st plot. Make sure this is the right one')
            if 'abs(probeScEta)<1.4442' in plot_dict[0]['cut'] and 'abs(probeScEta)>1.56' in plot_dict[1]['cut']:
                df_mc.loc[np.abs(df_mc['probeScEta'])<1.4442,'weight_clf'] = syst.utils.clf_reweight(df_mc.query('abs(probeScEta)<1.4442'), df_data, n_jobs=10, cut=plot_dict[0]['cut'])
                df_mc.loc[np.abs(df_mc['probeScEta'])>1.56,'weight_clf'] = syst.utils.clf_reweight(df_mc.query('abs(probeScEta)>1.56'), df_data, n_jobs=10, cut=plot_dict[1]['cut'])
            else:
                warnings.warn('Cut from 0th plot used to reweight whole dataset. Make sure this makes sense')
                df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=plot_dict[0]['cut'])

    #dont need this after earlier relabelling #NOTE: do need this for when it comes out of flashgg!
    if 'probePhiWidth' in varrs:
        df_data['probePhiWidth'] = df_data['probePhiWidth_Sc']

    if 'probeEtaWidth' in varrs:
        df_data['probeEtaWidth'] = df_data['probeEtaWidth_Sc']

    if 'probePhoIso03' in varrs:
        df_mc['probePhoIso'] = df_mc['probePhoIso03']
        df_data['probePhoIso'] = df_data['probePhoIso03']

    # if 'probePhoIso03' in varrs: #something wrong here?
    #    _ df_mc['probePhoIso03_uncorr'] = df_mc['probePhoIso_uncorr']

    if options.recomp_mva: #NOTE: need to do this for output of flashgg, since not sure xmls are updated to UL there
        stride = int(df_mc.index.size/10)
        #corrected vars have original names in fgg output
        correctedVariables = ['probeR9', 'probeS4', 'probeCovarianceIeIp', 'probeEtaWidth', 'probePhiWidth', 'probeSigmaIeIe', 'probePhoIso', 'probeChIso03', 'probeChIso03worst']
        weightsEB = "/vols/cms/jwd18/qRCSamples/IDMVAs/PhoID_barrel_UL2017_GJetMC_SATrain_nTree2k_LR_0p1_13052020_BDTG.weights.xml"
        weightsEE = "/vols/cms/jwd18/qRCSamples/IDMVAs/PhoID_endcap_UL2017_GJetMC_SATrain_nTree2k_LR_0p1_13052020_BDTG.weights.xml"
        if 'abs(probeScEta)<1.4442' in plot_dict[0]['cut']: df_mc['probeScPreshowerEnergy'] = np.zeros(df_mc.index.size)#NOTE: has a value in fgg for EE, which we read in, but not for EB
        df_mc['probePhoIdMVA_uncorr'] = np.concatenate(Parallel(n_jobs=10,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,df_mc[ch:ch+stride],'uncorr', False) for ch in range(0,df_mc.index.size,stride))) #uncorr
        df_mc['probePhoIdMVA_xml'] = np.concatenate(Parallel(n_jobs=10,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,df_mc[ch:ch+stride],'data', False) for ch in range(0,df_mc.index.size,stride))) #id from local xml
        #so df_mc['probePhoIdMVA'] is the one coming from flashgg
          

    varrs_miss = check_vars(df_mc, varrs)
    varrs_data_miss = check_vars(df_data, varrs_data)
    if any(varrs_miss + varrs_data_miss):
        print('Missing variables from mc df: ', list(itertools.compress(varrs,varrs_miss)))
        print('Missing variables from data df: ', list(itertools.compress(varrs_data,varrs_data_miss)))
        raise KeyError('Variables missing !')
    
    plots = []
    for dic in plot_dict:
        #plots.append(pldmc.plot_dmc_hist(df_mc, df_data=df_data, ratio=options.ratio, norm=options.norm, cut_str=options.cutstr, label=options.label, **dic))
        #NOTE: weight for plotting comes from clf_weight
        #plots.append(pldmc.plot_dmc_hist(df_mc, df_data=df_data, norm=options.norm, cut_str=options.cutstr, label=options.label, **dic))
        plot = pldmc.plot_dmc_hist(df_mc, df_data=df_data, norm=options.norm, cut_str=options.cutstr, label=options.label, **dic)
        plot.draw()
        plot.save(options.outdir, save_dill=options.save_dill)
        matplotlib.pyplot.close(plot.fig)

    #for plot in plots:
    #    plot.draw()
    #    plot.save(options.outdir, save_dill=options.save_dill)
    #    matplotlib.pyplot.close(plot.fig)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument('-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--data', action='store', type=str, required=True)
    requiredArgs.add_argument('-c', '--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-o', '--outdir', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    optionalArgs.add_argument('-r', '--ratio', action='store_true', default=False)
    optionalArgs.add_argument('-n', '--norm', action='store_true', default=False)
    optionalArgs.add_argument('-p', '--save_dill', action='store_true', default=False)
    optionalArgs.add_argument('-w', '--no_reweight', action='store_true', default=False)
    optionalArgs.add_argument('-k', '--cutstr', action='store_true', default=False)
    optionalArgs.add_argument('-M', '--recomp_mva', action='store_true', default=False)
    optionalArgs.add_argument('-l', '--label', action='store', type=str)
    optionalArgs.add_argument('-N', '--n_evts', action='store', type=int)
    optionalArgs.add_argument('-t', '--mc_tree', action='store', type=str)
    optionalArgs.add_argument('-s', '--data_tree', action='store', type=str)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    options = parser.parse_args()
    main(options)
